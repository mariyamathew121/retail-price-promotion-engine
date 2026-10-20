"""Build the reusable Module 1 retail analysis table from M5 source files."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

METADATA_COLUMNS = ("store_id", "state_id", "item_id", "dept_id", "cat_id")
PRICE_KEYS = ("store_id", "item_id", "wm_yr_wk")
SNAP_COLUMNS = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
PROMO_DISCOUNT_THRESHOLD = 0.10
REGULAR_PRICE_LOOKBACK = "56D"
OUTPUT_COLUMNS = (
    "date",
    "store_id",
    "state_id",
    "item_id",
    "dept_id",
    "cat_id",
    "units_sold",
    "price",
    "regular_price",
    "promo_flag",
    "discount_depth",
    "event_name",
    "event_type",
    "snap_flag",
    "weekday",
    "month",
)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("date", pa.timestamp("ns")),
        pa.field("store_id", pa.string()),
        pa.field("state_id", pa.string()),
        pa.field("item_id", pa.string()),
        pa.field("dept_id", pa.string()),
        pa.field("cat_id", pa.string()),
        pa.field("units_sold", pa.int32()),
        pa.field("price", pa.float64()),
        pa.field("regular_price", pa.float64()),
        pa.field("promo_flag", pa.int8()),
        pa.field("discount_depth", pa.float64()),
        pa.field("event_name", pa.string()),
        pa.field("event_type", pa.string()),
        pa.field("snap_flag", pa.int8()),
        pa.field("weekday", pa.string()),
        pa.field("month", pa.int8()),
    ]
)
DAY_COLUMN_PATTERN = re.compile(r"^d_(\d+)$")


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _add_rolling_regular_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Use the maximum observed price in the previous 8 calendar weeks."""
    prices = prices.sort_values(
        [*PRICE_KEYS[:2], "week_start"], kind="stable"
    ).reset_index(drop=True)
    rolling_max = (
        prices.groupby(list(PRICE_KEYS[:2]), sort=False)
        .rolling(REGULAR_PRICE_LOOKBACK, on="week_start", closed="left")[
            "sell_price"
        ]
        .max()
        .reset_index(level=list(range(len(PRICE_KEYS[:2]))), drop=True)
    )
    regular_prices = rolling_max.to_numpy().copy()
    has_price_history = np.isfinite(regular_prices)
    regular_prices[~has_price_history] = prices["sell_price"].to_numpy()[
        ~has_price_history
    ]
    prices["has_price_history"] = has_price_history
    prices["regular_price"] = regular_prices
    prices["discount_depth"] = (
        (prices["regular_price"] - prices["sell_price"])
        / prices["regular_price"]
    ).clip(lower=0)
    prices["promo_flag"] = (
        prices["has_price_history"]
        & (prices["discount_depth"] >= PROMO_DISCOUNT_THRESHOLD)
    ).astype("int8")
    return prices


def _prepare_calendar(calendar_path: Path) -> pd.DataFrame:
    calendar = pd.read_csv(calendar_path)
    required = {
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "month",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        *SNAP_COLUMNS.values(),
    }
    _require_columns(calendar, required, "calendar.csv")
    if calendar["d"].isna().any() or calendar["d"].duplicated().any():
        raise ValueError("calendar.csv must contain one non-empty row per day key")

    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    for column in ("event_name_1", "event_name_2", "event_type_1", "event_type_2"):
        calendar[column] = calendar[column].replace("", pd.NA)
    calendar["event_name"] = calendar["event_name_1"].combine_first(
        calendar["event_name_2"]
    )
    calendar["event_type"] = calendar["event_type_1"].combine_first(
        calendar["event_type_2"]
    )
    calendar["month"] = pd.to_numeric(calendar["month"], errors="raise").astype("int8")
    calendar["wm_yr_wk"] = pd.to_numeric(
        calendar["wm_yr_wk"], errors="raise"
    ).astype("int64")
    calendar["weekday"] = calendar["weekday"].astype("string")
    calendar["event_name"] = calendar["event_name"].astype("string")
    calendar["event_type"] = calendar["event_type"].astype("string")
    for column in SNAP_COLUMNS.values():
        calendar[column] = pd.to_numeric(calendar[column], errors="raise").astype("int8")

    return calendar


def _prepare_prices(prices_path: Path, calendar: pd.DataFrame) -> pd.DataFrame:
    prices = pd.read_csv(
        prices_path,
        dtype={"store_id": "string", "item_id": "string"},
    )
    _require_columns(prices, {*PRICE_KEYS, "sell_price"}, "sell_prices.csv")
    if prices[list(PRICE_KEYS)].isna().any().any():
        raise ValueError("sell_prices.csv contains missing price join keys")
    if prices.duplicated(list(PRICE_KEYS)).any():
        raise ValueError("sell_prices.csv has duplicate store/item/week keys")
    prices["wm_yr_wk"] = pd.to_numeric(
        prices["wm_yr_wk"], errors="raise"
    ).astype("int64")
    prices["sell_price"] = pd.to_numeric(prices["sell_price"], errors="raise")
    if (
        prices["sell_price"].isna().any()
        or not np.isfinite(prices["sell_price"].to_numpy()).all()
        or (prices["sell_price"] <= 0).any()
    ):
        raise ValueError("sell_prices.csv must contain finite, positive prices")

    week_starts = (
        calendar.groupby("wm_yr_wk", as_index=False, observed=True)["date"]
        .min()
        .rename(columns={"date": "week_start"})
    )
    prices = prices.merge(
        week_starts,
        on="wm_yr_wk",
        how="left",
        validate="many_to_one",
    )
    if prices["week_start"].isna().any():
        raise ValueError("calendar.csv is missing a week used by sell_prices.csv")
    return _add_rolling_regular_prices(prices)


def _day_columns(sales_path: Path) -> list[str]:
    header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    _require_columns(pd.DataFrame(columns=header), set(METADATA_COLUMNS), "sales file")
    days = [column for column in header if DAY_COLUMN_PATTERN.fullmatch(column)]
    days.sort(key=lambda column: int(DAY_COLUMN_PATTERN.fullmatch(column).group(1)))
    if not days:
        raise ValueError("Sales file does not contain any d_<number> columns")
    return days


def _sales_chunks(
    sales_path: Path, day_columns: list[str], chunk_skus: int
) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        sales_path,
        chunksize=chunk_skus,
        dtype={column: "string" for column in METADATA_COLUMNS},
        usecols=[*METADATA_COLUMNS, *day_columns],
    )


def _spot_check_rows(
    prices: pd.DataFrame, limit: int = 10
) -> pd.DataFrame:
    promo_counts = (
        prices.groupby(list(PRICE_KEYS[:2]), as_index=False, observed=True)[
            "promo_flag"
        ]
        .sum()
        .sort_values(
            ["promo_flag", *PRICE_KEYS[:2]],
            ascending=[False, True, True],
            kind="stable",
        )
    )
    preferred = promo_counts.loc[promo_counts["promo_flag"] > 0]
    selected = pd.concat(
        [
            preferred,
            promo_counts.loc[promo_counts["promo_flag"] == 0].sort_values(
                list(PRICE_KEYS[:2]), kind="stable"
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(list(PRICE_KEYS[:2])).head(limit)
    checked = prices.merge(
        selected[list(PRICE_KEYS[:2])],
        on=list(PRICE_KEYS[:2]),
        how="inner",
        validate="many_to_one",
    )
    return checked.rename(columns={"sell_price": "price"})[
        [
            "item_id",
            "store_id",
            "wm_yr_wk",
            "week_start",
            "price",
            "regular_price",
            "has_price_history",
            "promo_flag",
            "discount_depth",
        ]
    ].sort_values(["item_id", "store_id", "week_start"], kind="stable")


def _plot_spot_checks(spot_checks: pd.DataFrame, plot_path: Path) -> None:
    combinations = spot_checks[["item_id", "store_id"]].drop_duplicates()
    if combinations.empty:
        raise ValueError("No SKU-store combinations are available to plot")
    columns = 2
    rows = (len(combinations) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(16, max(4 * rows, 5)), squeeze=False
    )
    for axis, (_, combination) in zip(axes.flat, combinations.iterrows()):
        sample = spot_checks.loc[
            (spot_checks["item_id"] == combination["item_id"])
            & (spot_checks["store_id"] == combination["store_id"])
        ].sort_values("week_start")
        axis.plot(
            sample["week_start"],
            sample["regular_price"],
            color="#555555",
            linestyle="--",
            label="Prior 8-week max (regular)",
        )
        axis.plot(
            sample["week_start"],
            sample["price"],
            color="#2471A3",
            marker=".",
            label="Observed price",
        )
        promoted = sample["promo_flag"].astype(bool)
        if promoted.any():
            axis.scatter(
                sample.loc[promoted, "week_start"],
                sample.loc[promoted, "price"],
                color="#C0392B",
                marker="o",
                label="Flagged >=10% drop",
                zorder=3,
            )
        axis.set_title(f"{combination['item_id']} | {combination['store_id']}")
        axis.set_ylabel("Price")
        axis.grid(True, alpha=0.25)
        axis.tick_params(axis="x", labelrotation=30)
        axis.legend(fontsize="small")
    for axis in axes.flat[len(combinations) :]:
        axis.set_visible(False)
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _make_batch(
    chunk: pd.DataFrame,
    day_columns: list[str],
    calendar_arrays: dict[str, np.ndarray],
    price_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    values = chunk[day_columns].to_numpy(dtype=np.int32, copy=False)
    row_count, day_count = values.shape
    state_ids = chunk["state_id"].astype(str).to_numpy()
    unsupported = sorted(set(state_ids).difference(SNAP_COLUMNS))
    if unsupported:
        raise ValueError(
            "Unsupported state_id value(s) for M5 SNAP flags: "
            + ", ".join(unsupported)
        )
    state_snap_matrix = np.stack(
        [calendar_arrays[SNAP_COLUMNS[state]] for state in state_ids]
    )
    sales = pd.DataFrame(
        {
            "store_id": np.repeat(chunk["store_id"].to_numpy(), day_count),
            "state_id": np.repeat(chunk["state_id"].to_numpy(), day_count),
            "item_id": np.repeat(chunk["item_id"].to_numpy(), day_count),
            "dept_id": np.repeat(chunk["dept_id"].to_numpy(), day_count),
            "cat_id": np.repeat(chunk["cat_id"].to_numpy(), day_count),
            "d": np.tile(np.asarray(day_columns), row_count),
            "units_sold": values.reshape(-1),
            "date": np.tile(calendar_arrays["date"], row_count),
            "wm_yr_wk": np.tile(calendar_arrays["wm_yr_wk"], row_count),
            "weekday": np.tile(calendar_arrays["weekday"], row_count),
            "month": np.tile(calendar_arrays["month"], row_count),
            "event_name": np.tile(calendar_arrays["event_name"], row_count),
            "event_type": np.tile(calendar_arrays["event_type"], row_count),
            "snap_flag": state_snap_matrix.reshape(-1),
        }
    )

    day_price_keys = pd.MultiIndex.from_frame(sales.loc[:, list(PRICE_KEYS)])
    price_positions = price_lookup.index.get_indexer(day_price_keys)
    has_price = price_positions >= 0
    missing_price_rows = int((~has_price).sum())
    sales = sales.loc[has_price].copy()
    matched_prices = price_lookup.iloc[price_positions[has_price]]
    sales["price"] = matched_prices["sell_price"].to_numpy()
    sales["regular_price"] = matched_prices["regular_price"].to_numpy()
    sales["promo_flag"] = matched_prices["promo_flag"].to_numpy()
    sales["discount_depth"] = matched_prices["discount_depth"].to_numpy()
    sales = sales.loc[:, OUTPUT_COLUMNS]
    sales["units_sold"] = sales["units_sold"].astype("int32", copy=False)
    sales["promo_flag"] = sales["promo_flag"].astype("int8", copy=False)
    sales["snap_flag"] = sales["snap_flag"].astype("int8", copy=False)
    sales["month"] = sales["month"].astype("int8", copy=False)
    for column in ("store_id", "state_id", "item_id", "dept_id", "cat_id"):
        sales[column] = sales[column].astype("string")
    return sales, missing_price_rows


def build_clean_retail_data(
    sales_path: Path,
    prices_path: Path,
    calendar_path: Path,
    output_path: Path,
    *,
    chunk_skus: int = 128,
) -> dict[str, object]:
    """Build a single compressed Parquet file and accompanying audit artifacts."""
    if chunk_skus < 1:
        raise ValueError("chunk_skus must be at least 1")
    for path in (sales_path, prices_path, calendar_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required M5 input file not found: {path}")

    days = _day_columns(sales_path)
    calendar = _prepare_calendar(calendar_path)
    missing_calendar_days = sorted(set(days).difference(calendar["d"]))
    if missing_calendar_days:
        raise ValueError(
            f"calendar.csv is missing {len(missing_calendar_days)} sales day key(s)"
        )
    # Keep calendar values aligned to the d_1, d_2, ... columns, not CSV row order.
    calendar_arrays = calendar.set_index("d").loc[days][
        [
            "date",
            "wm_yr_wk",
            "weekday",
            "month",
            "event_name",
            "event_type",
            *SNAP_COLUMNS.values(),
        ]
    ].to_dict(orient="list")
    calendar_arrays = {key: np.asarray(value) for key, value in calendar_arrays.items()}

    prices = _prepare_prices(prices_path, calendar)
    price_lookup = prices.set_index(list(PRICE_KEYS)).sort_index()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output_path.with_name(f"{output_path.stem}_audit.json")
    spot_check_path = output_path.with_name(f"{output_path.stem}_promo_spot_checks.csv")
    spot_check_plot_path = output_path.with_name(
        f"{output_path.stem}_promo_spot_checks.png"
    )
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp.parquet"
    )

    source_rows = 0
    output_rows = 0
    dropped_no_price = 0
    promo_rows = 0
    flagged_below_threshold_rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        for chunk in _sales_chunks(sales_path, days, chunk_skus):
            if chunk[list(METADATA_COLUMNS)].isna().any().any():
                raise ValueError("Sales file contains missing SKU/store metadata")
            source_rows += len(chunk)
            batch, dropped = _make_batch(chunk, days, calendar_arrays, price_lookup)
            dropped_no_price += dropped
            if batch.empty:
                continue
            table = pa.Table.from_pandas(
                batch, schema=OUTPUT_SCHEMA, preserve_index=False
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path, OUTPUT_SCHEMA, compression="snappy"
                )
            writer.write_table(table, row_group_size=65_536)
            output_rows += len(batch)
            promo_rows += int(batch["promo_flag"].sum())
            flagged_below_threshold_rows += int(
                (
                    (batch["promo_flag"] == 1)
                    & (batch["discount_depth"] < PROMO_DISCOUNT_THRESHOLD)
                ).sum()
            )
        if writer is None:
            raise ValueError("No active SKU-day rows were found; no Parquet file was written")
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        temporary_path.replace(output_path)

    spot_checks = _spot_check_rows(prices)
    spot_checks.to_csv(spot_check_path, index=False)
    _plot_spot_checks(spot_checks, spot_check_plot_path)
    audit: dict[str, object] = {
        "source_files": {
            "sales": str(sales_path),
            "sell_prices": str(prices_path),
            "calendar": str(calendar_path),
        },
        "output_file": str(output_path),
        "source_sku_store_rows": source_rows,
        "sales_day_columns": len(days),
        "daily_rows_before_price_filter": source_rows * len(days),
        "output_active_sku_day_rows": output_rows,
        "dropped_rows_without_weekly_price": dropped_no_price,
        "active_rows_missing_price_after_filter": 0,
        "promo_flag_rows": promo_rows,
        "promo_share": promo_rows / output_rows,
        "promo_threshold": PROMO_DISCOUNT_THRESHOLD,
        "flagged_rows_with_discount_below_threshold": flagged_below_threshold_rows,
        "promo_flag_logic": (
            "promo_flag=1 only when discount_depth >= 0.10 and a previous "
            "price exists in the prior 8 calendar weeks; otherwise 0."
        ),
        "regular_price_logic": (
            "Maximum observed sell_price in the previous 8 calendar weeks, "
            "excluding the current week; if there is no previous price history, "
            "regular_price is the current sell_price."
        ),
        "spot_check_file": str(spot_check_path),
        "spot_check_plot_file": str(spot_check_plot_path),
        "spot_check_sku_store_combinations": int(
            spot_checks[["item_id", "store_id"]].drop_duplicates().shape[0]
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create clean_retail_data.parquet from the M5 input files."
    )
    parser.add_argument(
        "--sales",
        type=Path,
        default=Path("data/raw/sales_train_evaluation.csv"),
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("data/raw/sell_prices.csv"),
    )
    parser.add_argument(
        "--calendar",
        type=Path,
        default=Path("data/raw/calendar.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/clean_retail_data.parquet"),
    )
    parser.add_argument(
        "--chunk-skus",
        type=int,
        default=128,
        help="SKU-store rows to reshape at a time (lower this to reduce memory use).",
    )
    arguments = parser.parse_args()
    result = build_clean_retail_data(
        arguments.sales,
        arguments.prices,
        arguments.calendar,
        arguments.output,
        chunk_skus=arguments.chunk_skus,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
