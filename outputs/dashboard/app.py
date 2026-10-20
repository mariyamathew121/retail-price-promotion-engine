"""Local, data-backed dashboard for the retail decision-engine modules."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "blue": "#4466E8",
    "purple": "#805AD5",
    "orange": "#F28C45",
    "pink": "#E85D9E",
    "red": "#D94B55",
    "ink": "#20243A",
    "muted": "#6C7285",
}

MODULE_LIMITATIONS = {
    "Module 1": (
        "M5 contains no promotion flag. The price-drop proxy is inferred, "
        "and a missing price is treated as not active or not price-observable."
    ),
    "Module 2": (
        "Elasticities are observational. The pooled pricing defaults are "
        "department-level, not SKU-specific, and should prioritize experiments "
        "rather than trigger automatic price changes."
    ),
    "Module 3": (
        "Promotion windows are inferred from price drops, not retailer campaign "
        "records. Margins are assumptions; persistent price cuts and sparse "
        "baseline sales can affect ROI."
    ),
    "Module 4": (
        "D2 is synthetic and does not document sales periods or inventory "
        "timing. Markdown and profit figures are scenarios under assumptions, "
        "not validated retail recommendations."
    ),
    "Module 5": (
        "Instacart has no prices. Co-purchase is not cross-price response or "
        "causal bundle uplift; validate before pricing or promotion action."
    ),
}


def _file(relative_path: str) -> Path:
    return ROOT / relative_path


@st.cache_data(show_spinner=False)
def load_csv(relative_path: str) -> pd.DataFrame | None:
    path = _file(relative_path)
    if not path.is_file():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(relative_path: str) -> dict[str, Any] | None:
    path = _file(relative_path)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def card(label: str, value: str, detail: str = "", tone: str = "blue") -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_detail = html.escape(detail)
    st.markdown(
        f"""
        <div class="metric-card {tone}">
          <div class="metric-label">{safe_label}</div>
          <div class="metric-value">{safe_value}</div>
          <div class="metric-detail">{safe_detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cards(items: list[tuple[str, str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, detail, tone) in zip(columns, items):
        with column:
            card(label, value, detail, tone)


def source_note(paths: list[str]) -> None:
    readable = ", ".join(f"`{path}`" for path in paths)
    st.caption(f"Source artifacts: {readable}")


def limitation(module: str) -> None:
    st.warning(f"**Limitation — {module}:** {MODULE_LIMITATIONS[module]}")


def bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    *,
    color_by_sign: bool = False,
    height: int = 300,
) -> None:
    if data.empty:
        st.info("No chartable rows in this output.")
        return
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X(f"{x}:N", sort="-y", title=None),
        y=alt.Y(f"{y}:Q", title=None),
        tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=",.3f")],
    )
    if color_by_sign:
        bars = bars.encode(
            color=alt.condition(
                alt.datum[y] < 0,
                alt.value(COLORS["red"]),
                alt.value(COLORS["blue"]),
            )
        )
    else:
        bars = bars.encode(color=alt.value(color))
    st.altair_chart(
        bars.properties(height=height).configure_view(stroke=None),
        use_container_width=True,
    )


def missing_file(path: str) -> None:
    st.info(f"Output not found yet: `{path}`. Run the module pipeline to populate it.")


def format_number(value: Any, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def module1_page() -> None:
    st.subheader("Data foundation")
    audit_path = "data/processed/clean_retail_data_audit.json"
    audit = load_json(audit_path)
    spot_path = "data/processed/clean_retail_data_promo_spot_checks.csv"
    spot_checks = load_csv(spot_path)
    if audit is None:
        missing_file(audit_path)
    else:
        rows = int(audit.get("output_active_sku_day_rows", 0))
        dropped = int(audit.get("dropped_rows_without_weekly_price", 0))
        cards(
            [
                ("Active item-store-days", format_number(rows), "M5 output rows", "blue"),
                ("Rows without weekly price", format_number(dropped), "Excluded and audited", "orange"),
                ("Active rows missing price", format_number(audit.get("active_rows_missing_price_after_filter", 0)), "Expected: 0", "purple"),
                ("Inferred promo share", f"{float(audit.get('promo_share', 0)):.1%}", "Proxy, not source flag", "pink"),
            ]
        )
        st.caption(
            f"Source: {audit.get('source_sku_store_rows', '—'):,} item-store rows; "
            f"{audit.get('sales_day_columns', '—'):,} daily sales columns. "
            f"Promo threshold: {audit.get('promo_threshold', '—')}."
        )

    st.markdown("#### Promotion-proxy spot checks")
    if spot_checks is None:
        missing_file(spot_path)
    elif spot_checks.empty:
        st.info("Spot-check file is present but contains no rows.")
    else:
        choices = (
            spot_checks[["item_id", "store_id"]]
            .drop_duplicates()
            .astype(str)
            .apply(lambda row: f"{row.item_id} · {row.store_id}", axis=1)
            .tolist()
        )
        selected = st.selectbox("SKU-store sample", choices, key="m1_pair")
        item_id, store_id = selected.split(" · ", maxsplit=1)
        sample = spot_checks.loc[
            spot_checks["item_id"].astype(str).eq(item_id)
            & spot_checks["store_id"].astype(str).eq(store_id)
        ].copy()
        if "week_start" in sample:
            sample = sample.sort_values("week_start")
            sample["week_start"] = pd.to_datetime(
                sample["week_start"]
            ).dt.strftime("%Y-%m-%d")
            figure, axis = plt.subplots(figsize=(10, 3))
            axis.plot(
                sample["week_start"],
                sample["price"],
                color=COLORS["blue"],
                linewidth=2,
                label="Observed price",
            )
            axis.plot(
                sample["week_start"],
                sample["regular_price"],
                color=COLORS["purple"],
                linewidth=2,
                label="Prior-window regular price",
            )
            axis.set_ylabel("Price")
            axis.grid(axis="y", color="#E3E7F2", linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False, ncol=2)
            figure.autofmt_xdate()
            st.pyplot(figure)
            plt.close(figure)
        st.dataframe(sample, use_container_width=True, hide_index=True)
    source_note(
        [
            audit_path,
            spot_path,
            "data/processed/clean_retail_data.parquet",
            "outputs/module1/query_benchmark.md",
        ]
    )
    limitation("Module 1")


def module2_page() -> None:
    st.subheader("Price elasticity")
    series_path = "data/processed/module2/price_elasticities.csv"
    pooled_path = "data/processed/module2_pooled/pooled_price_elasticities.csv"
    defaults_path = "data/processed/module2_pooled/item_store_pricing_defaults.csv"
    series = load_csv(series_path)
    pooled = load_csv(pooled_path)
    defaults = load_csv(defaults_path)
    if series is None:
        missing_file(series_path)
        limitation("Module 2")
        return

    estimated = series.loc[series.get("fit_status", pd.Series(index=series.index)).eq("estimated")]
    positive_share = (
        float(estimated["elasticity"].gt(0).mean()) if len(estimated) else float("nan")
    )
    cards(
        [
            ("Selected SKU-store series", format_number(len(series)), "Per-series DML and OLS", "blue"),
            ("Estimated", format_number(len(estimated)), "Rows with fitted model", "purple"),
            ("Positive DML estimates", f"{positive_share:.1%}" if pd.notna(positive_share) else "—", "Flag for review", "orange"),
            ("Pooled defaults", format_number(len(defaults)) if defaults is not None else "—", "Department estimate attached to SKU-store", "pink"),
        ]
    )
    if pooled is not None:
        department = pooled.loc[pooled["store_id"].astype(str).eq("ALL")].copy()
        if not department.empty:
            st.markdown("#### Department-level pooled PPML estimates")
            st.caption("95% confidence intervals shown; classification uses the upper bound.")
            department = department.sort_values("ppml_elasticity")
            department_order = department["dept_id"].tolist()
            interval_chart = (
                alt.Chart(department)
                .mark_rule(color=COLORS["blue"], strokeWidth=3)
                .encode(
                    x=alt.X("dept_id:N", sort=department_order, title=None),
                    y=alt.Y("ppml_ci_lower:Q", title="PPML elasticity (95% CI)"),
                    y2=alt.Y2("ppml_ci_upper:Q"),
                    tooltip=[
                        "dept_id:N",
                        alt.Tooltip("ppml_elasticity:Q", format=".3f"),
                        alt.Tooltip("ppml_ci_lower:Q", format=".3f"),
                        alt.Tooltip("ppml_ci_upper:Q", format=".3f"),
                    ],
                )
            )
            point_chart = (
                alt.Chart(department)
                .mark_point(filled=True, size=90, color=COLORS["purple"])
                .encode(
                    x=alt.X("dept_id:N", sort=department_order, title=None),
                    y="ppml_elasticity:Q",
                )
            )
            st.altair_chart(
                (interval_chart + point_chart)
                .properties(height=320)
                .configure_view(stroke=None),
                use_container_width=True,
            )
            st.dataframe(
                department[
                    [
                        "dept_id",
                        "selected_series",
                        "ppml_elasticity",
                        "ppml_ci_lower",
                        "ppml_ci_upper",
                        "ppml_p_value",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
    st.markdown("#### Per-series results")
    display_columns = [
        column
        for column in [
            "item_id",
            "store_id",
            "elasticity",
            "ci_lower",
            "ci_upper",
            "naive_elasticity",
            "ols_elasticity",
            "demand_type",
            "pricing_recommendation",
        ]
        if column in series.columns
    ]
    st.dataframe(
        series[display_columns].sort_values("elasticity").head(30),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("#### Interpretation")
    interpretation_path = "outputs/module2/naive_dml_pooled_comparison.md"
    interpretation = _file(interpretation_path)
    if interpretation.is_file():
        st.markdown(interpretation.read_text(encoding="utf-8"))
    source_note([series_path, pooled_path, defaults_path, interpretation_path])
    limitation("Module 2")


def module3_page() -> None:
    st.subheader("Promotion uplift")
    path = "data/processed/module3/promotion_uplift.csv"
    data = load_csv(path)
    if data is None:
        missing_file(path)
        limitation("Module 3")
        return

    complete = data.loc[data["analysis_status"].eq("complete")]
    persistent = data["price_persists"].astype(str).str.lower().eq("true")
    zero_baseline_share = (
        float(complete["baseline_sales"].eq(0).mean()) if len(complete) else float("nan")
    )
    no_increment = (
        float(complete["promo_sales"].le(complete["baseline_sales"]).mean())
        if len(complete)
        else float("nan")
    )
    cards(
        [
            ("Promotion episodes", format_number(len(data)), "Inferred price-drop windows", "blue"),
            ("Complete episodes", format_number(len(complete)), "Baseline available", "purple"),
            ("Possible persistent cuts", format_number(persistent.sum()), "Review; may not be temporary promos", "orange"),
            ("Zero baseline / promo ≤ baseline", f"{zero_baseline_share:.1%} / {no_increment:.1%}", "Among complete episodes", "pink"),
        ]
    )
    recommendation_counts = data["recommendation"].value_counts().rename_axis("recommendation")
    st.markdown("#### Recommendation counts")
    rec_chart = recommendation_counts.rename("episodes").reset_index()
    bar_chart(rec_chart, "recommendation", "episodes", COLORS["purple"])
    category_roi = (
        complete.groupby("cat_id", dropna=False)["promotion_ROI"]
        .median()
        .sort_values()
        .rename("median_ROI")
    )
    st.markdown("#### Median promotion ROI by category")
    if not category_roi.empty:
        roi_chart = category_roi.rename_axis("category").reset_index()
        bar_chart(
            roi_chart,
            "category",
            "median_ROI",
            COLORS["orange"],
            color_by_sign=True,
        )
    depth_roi = complete.assign(
        depth_band=pd.cut(
            complete["avg_discount_depth"],
            [0.10, 0.15, 0.25, 0.40, 1.0],
            include_lowest=False,
        )
    )
    st.markdown("#### ROI by discount depth")
    depth_summary = (
        depth_roi.groupby("depth_band", observed=False)["promotion_ROI"]
        .agg(episodes="count", median_ROI="median")
        .reset_index()
    )
    st.dataframe(depth_summary, use_container_width=True, hide_index=True)
    st.markdown("#### Episode detail")
    st.dataframe(
        data.sort_values("promotion_ROI", ascending=True, na_position="last").head(100),
        use_container_width=True,
        hide_index=True,
    )
    tlearner_path = "data/processed/module3_causalml/promotion_tlearner_observational.csv"
    tlearner = load_csv(tlearner_path)
    if tlearner is not None:
        st.markdown("#### Observational CausalML diagnostic")
        st.caption(
            "T-learner summaries use inferred price-drop exposure; they are "
            "diagnostics, not causal uplift estimates or promotion recommendations."
        )
        st.dataframe(
            tlearner.sort_values(
                "mean_predicted_treatment_effect_units",
                ascending=False,
            ).head(30),
            use_container_width=True,
            hide_index=True,
        )
    source_note([
        path,
        "data/processed/module3/promotion_uplift_audit.json",
        tlearner_path,
        "data/processed/module3_causalml/promotion_tlearner_observational_audit.json",
        "outputs/module3/promotion_uplift_summary.md",
    ])
    limitation("Module 3")


def module4_page() -> None:
    st.subheader("Markdown scenarios")
    path = "data/processed/module4/markdown_recommendations.csv"
    data = load_csv(path)
    audit_path = "data/processed/module4/markdown_recommendations_audit.json"
    audit = load_json(audit_path)
    sensitivity_path = "outputs/module4/evidence/markdown_period_horizon_sensitivity.csv"
    sensitivity = load_csv(sensitivity_path)
    if data is None:
        missing_file(path)
        limitation("Module 4")
        return
    if audit is not None:
        period = audit.get("historical_sales_period_weeks", "—")
        horizon = audit.get("horizon_weeks", "—")
    else:
        period = int(data["historical_sales_period_weeks"].mode().iloc[0])
        horizon = int(data["horizon_weeks"].mode().iloc[0])
    markdown_distribution = (
        data["recommended_markdown"].value_counts().sort_index()
    )
    below_horizon = data["weeks_cover"].le(horizon)
    cards(
        [
            ("Products", format_number(len(data)), "Scenario output rows", "blue"),
            ("Sales-period assumption", f"{period} weeks", "Illustrative; not documented by D2", "purple"),
            ("Horizon", f"{horizon} weeks", "Scenario parameter", "orange"),
            ("At 0% markdown", format_number(markdown_distribution.get(0.0, 0)), "Markdown grid: 0–50%", "pink"),
        ]
    )
    st.markdown("#### Recommended markdown distribution")
    distribution = markdown_distribution.rename_axis("markdown").rename("products").reset_index()
    distribution["markdown_label"] = distribution["markdown"].map(
        lambda value: f"{value:.0%}"
    )
    bar_chart(distribution, "markdown_label", "products", COLORS["blue"])
    if "action" in data.columns:
        st.markdown("#### Business-rule actions")
        actions = data["action"].value_counts().rename_axis("action").rename("products").reset_index()
        bar_chart(actions, "action", "products", COLORS["purple"])
    st.markdown("#### Scenario profit summary")
    cards(
        [
            ("Median expected margin", f"${data['expected_margin'].median():,.2f}", "Cost and salvage are assumptions", "purple"),
            ("Median stock cover", f"{data['weeks_cover'].replace([float('inf')], pd.NA).median():.2f} weeks", "Stock ÷ assumed weekly sales", "blue"),
            ("Cover ≤ horizon", f"{below_horizon.mean():.1%}", "Policy holds at 0% markdown", "orange"),
            ("Markdown-eligible", format_number(data["markdown_eligible_by_weeks_cover"].astype(str).str.lower().eq("true").sum()), "Strictly greater cover than horizon", "pink"),
        ]
    )
    if sensitivity is not None:
        st.markdown("#### Sales-period / horizon sensitivity")
        st.dataframe(sensitivity, use_container_width=True, hide_index=True)
    st.markdown("#### Example SKU scenarios")
    examples = data.sort_values("profit_gain_vs_fixed_30", ascending=False).head(50)
    st.dataframe(examples, use_container_width=True, hide_index=True)
    source_note(
        [
            path,
            audit_path,
            sensitivity_path,
            "outputs/module4/evidence/sku_profit_curves.csv",
        ]
    )
    limitation("Module 4")


def module5_page() -> None:
    st.subheader("Bundles and cross-sell candidates")
    path = "data/processed/module5/bundle_candidates.csv"
    candidates = load_csv(path)
    overlap_path = "data/processed/module5/fpgrowth_top50_overlap.csv"
    overlap = load_csv(overlap_path)
    if candidates is None:
        missing_file(path)
        limitation("Module 5")
        return
    mapped = candidates["m5_category"].fillna("").ne("")
    top_overlap = (
        int(overlap["in_bundle_candidates"].astype(str).str.lower().eq("true").sum())
        if overlap is not None
        else None
    )
    cards(
        [
            ("Bundle candidates", format_number(len(candidates)), "Prior support ≥ 0.25%; lift > 1.5", "blue"),
            ("Mapped to M5", format_number(mapped.sum()), "Department crosswalk assumption", "purple"),
            ("Unmapped", format_number((~mapped).sum()), "Blank when no same-category mapping", "orange"),
            ("FP-Growth top-50 overlap", f"{top_overlap}/50" if top_overlap is not None else "—", "100k-order sample comparison", "pink"),
        ]
    )
    department_pairs = pd.concat(
        [
            candidates[["item_a", "item_b", "department_a"]].rename(
                columns={"department_a": "department"}
            ),
            candidates[["item_a", "item_b", "department_b"]].rename(
                columns={"department_b": "department"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(["item_a", "item_b", "department"])
    department_counts = department_pairs["department"].value_counts().sort_values()
    st.markdown("#### Candidate coverage by department")
    department_chart = department_counts.rename_axis("department").rename("pairs").reset_index()
    bar_chart(department_chart, "department", "pairs", COLORS["purple"])
    st.markdown("#### Highest-lift candidate pairs")
    st.dataframe(
        candidates.head(50)[
            [
                "product_a_name",
                "department_a",
                "product_b_name",
                "department_b",
                "prior_support",
                "confidence_a_to_b",
                "confidence_b_to_a",
                "prior_lift",
                "train_lift",
                "m5_category",
                "validation_status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    if overlap is not None:
        st.markdown("#### FP-Growth sample top 50")
        st.dataframe(overlap, use_container_width=True, hide_index=True)
    source_note(
        [
            path,
            "data/processed/module5/bundle_candidate_summary.md",
            overlap_path,
        ]
    )
    limitation("Module 5")


def main() -> None:
    st.set_page_config(
        page_title="Retail Decision Engine",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        f"""
        <style>
        :root {{
          --primary-color: {COLORS["blue"]};
          --background-color: #F5F7FC;
          --secondary-background-color: #FFFFFF;
          --text-color: {COLORS["ink"]};
        }}
        .stApp {{ background: #F5F7FC; color: {COLORS["ink"]}; }}
        [data-testid="stHeader"] {{ background: rgba(245,247,252,0.92); }}
        [data-testid="stSidebar"] {{ background: #EEF1FA; }}
        .block-container {{ padding-top: 1.5rem; max-width: 1500px; }}
        .metric-card {{
          background: #FFFFFF;
          border: 1px solid #E3E7F2;
          border-top: 4px solid {COLORS["blue"]};
          border-radius: 14px;
          padding: 16px 18px;
          min-height: 122px;
          box-shadow: 0 3px 12px rgba(34,46,88,0.05);
        }}
        .metric-card.purple {{ border-top-color: {COLORS["purple"]}; }}
        .metric-card.orange {{ border-top-color: {COLORS["orange"]}; }}
        .metric-card.pink {{ border-top-color: {COLORS["pink"]}; }}
        .metric-card.red {{ border-top-color: {COLORS["red"]}; }}
        .metric-label {{ color: {COLORS["muted"]}; font-size: 0.84rem; font-weight: 650; }}
        .metric-value {{ color: {COLORS["ink"]}; font-size: 1.65rem; font-weight: 760; margin-top: 8px; }}
        .metric-detail {{ color: {COLORS["muted"]}; font-size: 0.75rem; margin-top: 4px; }}
        h1, h2, h3 {{ color: {COLORS["ink"]}; letter-spacing: -0.02em; }}
        h1 {{ font-size: clamp(1.7rem, 4vw, 2.8rem) !important; line-height: 1.15; }}
        [data-testid="stMetric"] {{ background: white; border: 1px solid #E3E7F2; padding: 14px; border-radius: 12px; }}
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {{ border-color: {COLORS["blue"]}; background: #FFFFFF; }}
        [data-testid="stSelectbox"] [data-baseweb="select"] div,
        [data-testid="stSelectbox"] input[role="combobox"] {{
          color: {COLORS["ink"]} !important;
          -webkit-text-fill-color: {COLORS["ink"]} !important;
        }}
        div[data-testid="stAlert"] {{ border-radius: 12px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Retail Price & Promotion Decision Engine")
    st.caption(
        "Traceable analysis workspace · M5 evidence and scenario outputs · "
        "Use recommendations as review and experiment inputs, not automatic actions."
    )
    with st.sidebar:
        st.markdown("### Project status")
        st.markdown("**5 analysis modules**")
        st.caption("Each page shows its source artifacts and a module-specific limitation.")
        st.markdown("---")
        st.caption("Palette: blue · purple · orange · pink · red")

    modules = {
        "1 · Data": module1_page,
        "2 · Price": module2_page,
        "3 · Promotions": module3_page,
        "4 · Markdowns": module4_page,
        "5 · Bundles": module5_page,
    }
    selected_module = st.selectbox("Module", list(modules), label_visibility="collapsed")
    modules[selected_module]()


if __name__ == "__main__":
    main()
