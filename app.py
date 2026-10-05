import os
import streamlit as st
import pandas as pd

from quality_engine import analyze_data
from ai_engine import (
    calculate_readiness,
    prioritize_issues,
    investigate_issue,
    project_impact,
    generate_audit_entry,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ReportGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 15px;
    }

    .trust-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background-color: #f8fafc;
    }

    .ai-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dbeafe;
        background-color: #eff6ff;
    }

    .demo-box {
        padding: 14px;
        border-radius: 10px;
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
    }

    div[data-testid="stMetric"] {
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ ReportGuard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Find the problems that matter before a report goes out."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Analysis")

    demo_mode = st.checkbox(
        "🎬 Demo Mode",
        value=False,
    )

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx"],
    )

    analyze_button = st.button(
        "▶ Run Analysis",
        use_container_width=True,
    )

    st.divider()

    if demo_mode:

        st.markdown(
            '<div class="demo-box">'
            "<b>🎬 Demo Mode Active</b><br>"
            "Using the prepared corrupted dataset."
            "</div>",
            unsafe_allow_html=True,
        )

    else:

        st.caption(
            "Upload a CSV or XLSX dataset to begin."
        )

    st.divider()

    st.caption("ReportGuard")
    st.caption(
        "Detect → Prioritize → Investigate → Explain"
    )


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    records_placeholder = st.empty()
    records_placeholder.metric(
        "Records",
        "—"
    )

with col2:
    issues_placeholder = st.empty()
    issues_placeholder.metric(
        "Issues",
        "—"
    )

with col3:
    readiness_placeholder = st.empty()
    readiness_placeholder.metric(
        "Readiness",
        "—"
    )

with col4:
    impact_placeholder = st.empty()
    impact_placeholder.metric(
        "High Impact",
        "—"
    )


# ============================================================
# INITIAL STATE
# ============================================================

df = None
results = None
summary = None
ranked_issues = []
ranked_clusters = []


# ============================================================
# LOAD DATA
# ============================================================

try:

    # --------------------------------------------------------
    # DEMO MODE
    # --------------------------------------------------------

    if demo_mode:

        demo_path = os.path.join(
            os.path.dirname(__file__),
            "demo_corrupted.xlsx",
        )

        if os.path.exists(demo_path):

            df = pd.read_excel(
                demo_path
            )

            st.success(
                f"🎬 Demo dataset loaded — "
                f"**{len(df):,} records**"
            )

        else:

            st.error(
                "Demo dataset not found. "
                "Place `demo_corrupted.xlsx` "
                "in the same folder as `app.py`."
            )


    # --------------------------------------------------------
    # NORMAL UPLOAD
    # --------------------------------------------------------

    elif uploaded_file is not None:

        if uploaded_file.name.lower().endswith(
            ".csv"
        ):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

        st.success(
            f"Loaded **{uploaded_file.name}** "
            f"— **{len(df):,} records**"
        )


    # ========================================================
    # ANALYSIS
    # ========================================================

    if df is not None:

        # ----------------------------------------------------
        # QUALITY ENGINE
        # ----------------------------------------------------

        results = analyze_data(
            df
        )

        summary = results[
            "summary"
        ]


        # ----------------------------------------------------
        # PRIORITIZATION
        # ----------------------------------------------------

        ranked_issues = prioritize_issues(
            results["clusters"]
        )

        ranked_clusters = [
            item["original_issue"]
            for item in ranked_issues
        ]


        # ----------------------------------------------------
        # READINESS
        # ----------------------------------------------------

        readiness = calculate_readiness(
            results
        )


        # ----------------------------------------------------
        # TOP METRICS
        # ----------------------------------------------------

        records_placeholder.metric(
            "Records",
            f"{len(df):,}"
        )

        issues_placeholder.metric(
            "Issues",
            f"{summary['raw_issue_count']:,}"
        )

        readiness_placeholder.metric(
            "Readiness",
            f"{readiness['score']}%"
        )

        impact_placeholder.metric(
            "High Impact",
            summary["critical"]
            + summary["high"]
        )


        # ====================================================
        # EXECUTIVE SUMMARY
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            "📊 Executive Summary"
            "</div>",
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Records",
                f"{len(df):,}"
            )

        with col2:

            st.metric(
                "Raw Issues",
                f"{summary['raw_issue_count']:,}"
            )

        with col3:

            st.metric(
                "Critical",
                summary["critical"]
            )

        with col4:

            st.metric(
                "High",
                summary["high"]
            )

        st.caption(
            f"{summary['raw_issue_count']:,} raw problems "
            f"were grouped into "
            f"{summary['issue_groups']} issue groups."
        )


        # ====================================================
        # ISSUE INTELLIGENCE
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            "🎯 Issue Intelligence"
            "</div>",
            unsafe_allow_html=True,
        )

        if ranked_clusters:

            for i, cluster in enumerate(
                ranked_clusters[:5]
            ):

                severity = cluster[
                    "severity"
                ]

                if severity == "Critical":

                    badge = "🔴 CRITICAL"

                elif severity == "High":

                    badge = "🟠 HIGH"

                elif severity == "Medium":

                    badge = "🟡 MEDIUM"

                else:

                    badge = "🟢 LOW"

                with st.container(
                    border=True
                ):

                    col1, col2, col3 = (
                        st.columns(
                            [3, 1, 1]
                        )
                    )

                    with col1:

                        st.subheader(
                            cluster[
                                "cluster_name"
                            ]
                        )

                    with col2:

                        st.markdown(
                            f"**{badge}**"
                        )

                    with col3:

                        st.metric(
                            "Affected",
                            f"{cluster['affected_rows']:,}"
                        )

                    st.caption(
                        "Detection confidence: "
                        f"{cluster['confidence']:.0%}"
                    )

                    if st.button(
                        "🔍 Investigate",
                        key=f"investigate_{i}",
                        use_container_width=True,
                    ):

                        st.session_state[
                            "selected_cluster"
                        ] = cluster

        else:

            st.success(
                "No significant issue groups detected."
            )


        # ====================================================
        # INVESTIGATION
        # ====================================================

        selected_cluster = (
            st.session_state.get(
                "selected_cluster"
            )
        )

        if selected_cluster:

            investigation = investigate_issue(
                selected_cluster,
                summary,
            )

            st.divider()

            st.markdown(
                '<div class="section-title">'
                "🔎 Investigation"
                "</div>",
                unsafe_allow_html=True,
            )

            st.subheader(
                selected_cluster[
                    "cluster_name"
                ]
            )


            # ------------------------------------------------
            # ISSUE METRICS
            # ------------------------------------------------

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "Severity",
                    selected_cluster[
                        "severity"
                    ],
                )

            with col2:

                st.metric(
                    "Affected Records",
                    f"{selected_cluster['affected_rows']:,}",
                )

            with col3:

                st.metric(
                    "Confidence",
                    f"{selected_cluster['confidence']:.0%}",
                )


            # ------------------------------------------------
            # TRUST MESSAGE
            # ------------------------------------------------

            st.info(
                "🔎 **Detection** uses deterministic "
                "data-quality rules. "
                "🤖 **Investigation** uses AI-assisted "
                "interpretation when available. "
                "👤 **Final decisions** require human review."
            )


            # ------------------------------------------------
            # WHAT'S WRONG
            # ------------------------------------------------

            st.write(
                "### What's Wrong?"
            )

            evidence = selected_cluster.get(
                "evidence",
                []
            )

            if evidence:

                for item in evidence:

                    st.write(
                        f"• {item}"
                    )

            else:

                st.write(
                    "Evidence is not available."
                )


            # ------------------------------------------------
            # ROOT CAUSE
            # ------------------------------------------------

            st.write(
                "### Likely Root Cause"
            )

            st.write(
                selected_cluster.get(
                    "likely_root_cause",
                    "Not available.",
                )
            )


            # ------------------------------------------------
            # RECOMMENDATION
            # ------------------------------------------------

            st.write(
                "### Recommended Action"
            )

            st.write(
                selected_cluster.get(
                    "recommended_action",
                    "Not available.",
                )
            )


            # =================================================
            # AI INTERPRETATION
            # =================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                "🤖 AI Interpretation"
                "</div>",
                unsafe_allow_html=True,
            )

            source = investigation.get(
                "source",
                "Deterministic fallback",
            )

            if source == "llm":

                st.caption(
                    "AI-assisted interpretation"
                )

            else:

                st.caption(
                    "Deterministic fallback interpretation"
                )


            if investigation.get(
                "issue_summary"
            ):

                st.write(
                    "**Issue Summary**"
                )

                st.write(
                    investigation[
                        "issue_summary"
                    ]
                )


            if investigation.get(
                "interpretation"
            ):

                st.write(
                    "**Why It Matters**"
                )

                st.write(
                    investigation[
                        "interpretation"
                    ]
                )


            if investigation.get(
                "likely_root_cause"
            ):

                st.write(
                    "**AI-Assessed Root Cause**"
                )

                st.write(
                    investigation[
                        "likely_root_cause"
                    ]
                )


            if investigation.get(
                "recommendation"
            ):

                st.write(
                    "**AI Recommendation**"
                )

                st.write(
                    investigation[
                        "recommendation"
                    ]
                )


            if investigation.get(
                "confidence"
            ):

                st.write(
                    "**Investigation Confidence:** "
                    f"{investigation['confidence']}"
                )


            assumptions = (
                investigation.get(
                    "assumptions",
                    []
                )
            )

            if assumptions:

                st.write(
                    "**Assumptions**"
                )

                for assumption in assumptions:

                    st.write(
                        f"• {assumption}"
                    )


            # =================================================
            # PROJECTED IMPACT
            # =================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                "📈 Projected Impact"
                "</div>",
                unsafe_allow_html=True,
            )

            impact = project_impact(
                selected_cluster,
                results,
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "Current Readiness",
                    f"{impact.get('current_readiness', readiness['score'])}%",
                )

            with col2:

                st.metric(
                    "Projected Readiness",
                    f"{impact.get('projected_readiness', readiness['score'])}%",
                )

            with col3:

                st.metric(
                    "Potential Improvement",
                    f"+{impact.get('improvement', 0)}",
                )

            st.info(
                "Projected impact is an estimate, "
                "not a guarantee."
            )

            if impact.get("note"):

                st.caption(
                    impact["note"]
                )


            # =================================================
            # AUDIT & TRUST
            # =================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                "🔐 Audit & Trust"
                "</div>",
                unsafe_allow_html=True,
            )

            audit_entry = generate_audit_entry(
                selected_cluster,
                investigation,
            )

            col1, col2 = (
                st.columns(2)
            )

            with col1:

                st.markdown(
                    "**Detection Method**"
                )

                st.write(
                    "Deterministic quality rules"
                )

                st.markdown(
                    "**Evidence**"
                )

                st.write(
                    f"{selected_cluster['affected_rows']:,} "
                    "affected records"
                )

            with col2:

                st.markdown(
                    "**Investigation Method**"
                )

                if source == "llm":

                    st.write(
                        "AI-assisted investigation"
                    )

                else:

                    st.write(
                        "Deterministic fallback"
                    )

                st.markdown(
                    "**Human Review**"
                )

                st.write(
                    "Required before corrective action"
                )


            with st.expander(
                "View audit details"
            ):

                if isinstance(
                    audit_entry,
                    dict
                ):

                    for key, value in (
                        audit_entry.items()
                    ):

                        formatted_key = (
                            str(key)
                            .replace(
                                "_",
                                " "
                            )
                            .title()
                        )

                        st.write(
                            f"**{formatted_key}:**"
                        )

                        if isinstance(
                            value,
                            list
                        ):

                            for item in value:

                                st.write(
                                    f"• {item}"
                                )

                        else:

                            st.write(
                                str(value)
                            )

                else:

                    st.write(
                        audit_entry
                    )


        # ====================================================
        # RAW DATA
        # ====================================================

        st.divider()

        with st.expander(
            "📄 View Raw Data"
        ):

            st.dataframe(
                df.head(100),
                use_container_width=True,
            )

            st.caption(
                "Showing the first 100 records."
            )


except Exception as e:

    st.error(
        "ReportGuard could not process this dataset."
    )

    st.exception(e)


# ============================================================
# EMPTY STATE
# ============================================================

if (
    df is None
    and not demo_mode
    and uploaded_file is None
):

    st.info(
        "Upload a CSV/XLSX dataset or enable "
        "🎬 Demo Mode to begin."
    )