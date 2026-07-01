"""
EduPro — Instructor Performance & Course Quality Evaluation
Streamlit web application.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_pipeline import (
    load_raw, validate, build_master, instructor_summary,
    add_rating_tiers, compute_kpis,
)

st.set_page_config(
    page_title="EduPro | Instructor & Course Quality Analytics",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def get_data():
    teachers, courses, transactions, users = load_raw()
    master = build_master(teachers, courses, transactions)
    summary = add_rating_tiers(instructor_summary(master, teachers))
    return teachers, courses, transactions, users, master, summary


teachers, courses, transactions, users, master, summary = get_data()

TIER_ORDER = ["Low (<2.5)", "Mid (2.5–3.99)", "High (4.0+)"]
TIER_COLORS = {"Low (<2.5)": "#E74C3C", "Mid (2.5–3.99)": "#F5B041", "High (4.0+)": "#2ECC71"}

# ---------------------------------------------------------------------------
# Sidebar — global filters
# ---------------------------------------------------------------------------

st.sidebar.title("🎓 EduPro Analytics")
st.sidebar.caption("Instructor Performance & Course Quality Evaluation")
st.sidebar.markdown("---")

st.sidebar.subheader("Filters")

expertise_opts = sorted(teachers["Expertise"].unique())
sel_expertise = st.sidebar.multiselect("Instructor Expertise", expertise_opts, default=[])

category_opts = sorted(courses["CourseCategory"].unique())
sel_category = st.sidebar.multiselect("Course Category", category_opts, default=[])

level_opts = sorted(courses["CourseLevel"].unique())
sel_level = st.sidebar.multiselect("Course Level", level_opts, default=[])

gender_opts = sorted(teachers["Gender"].unique())
sel_gender = st.sidebar.multiselect("Instructor Gender", gender_opts, default=[])

st.sidebar.markdown("**Teacher Rating range**")
teacher_rating_range = st.sidebar.slider(
    "Filter instructors by rating", 1.0, 5.0, (1.0, 5.0), step=0.1, key="tr_range"
)

st.sidebar.markdown("**Course Rating range**")
course_rating_range = st.sidebar.slider(
    "Filter courses by rating", 1.0, 5.0, (1.0, 5.0), step=0.1, key="cr_range"
)

exp_min, exp_max = int(teachers.YearsOfExperience.min()), int(teachers.YearsOfExperience.max())
exp_range = st.sidebar.slider("Years of Experience", exp_min, exp_max, (exp_min, exp_max))


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all sidebar filters to a transaction-level (master-joined) dataframe."""
    out = df.copy()
    if sel_expertise:
        out = out[out["Expertise"].isin(sel_expertise)]
    if sel_category:
        out = out[out["CourseCategory"].isin(sel_category)]
    if sel_level:
        out = out[out["CourseLevel"].isin(sel_level)]
    if sel_gender:
        out = out[out["Gender"].isin(sel_gender)]
    out = out[
        (out["TeacherRating"].between(*teacher_rating_range)) &
        (out["CourseRating"].between(*course_rating_range)) &
        (out["YearsOfExperience"].between(*exp_range))
    ]
    return out


filtered_master = apply_filters(master)

# Rebuild a filtered instructor summary consistent with the active filters
filtered_teacher_ids = filtered_master["TeacherID"].unique()
filtered_teachers = teachers[teachers["TeacherID"].isin(filtered_teacher_ids)]
filtered_summary = add_rating_tiers(instructor_summary(filtered_master, filtered_teachers))

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered_master):,}** of **{len(master):,}** enrollment records")
st.sidebar.caption(f"**{filtered_teachers.shape[0]}** of **{teachers.shape[0]}** instructors match filters")

with st.sidebar.expander("ℹ️ About the data model"):
    st.write(
        "Each course in this dataset is delivered by **multiple instructors** "
        "across different enrollments, and each instructor teaches **multiple "
        "courses**. There is no fixed 1:1 'course owner'. All analysis below is "
        "therefore computed at the **enrollment (transaction) level** — joining "
        "Teachers ↔ Transactions ↔ Courses — which is the only level at which "
        "instructor and course performance can be correctly attributed."
    )

# ---------------------------------------------------------------------------
# Header & KPIs
# ---------------------------------------------------------------------------

st.title("Instructor Performance & Course Quality Evaluation")
st.caption("EduPro · Data-driven framework for instructor effectiveness and course quality consistency")

if filtered_master.empty:
    st.warning("No data matches the current filters. Try widening your selection in the sidebar.")
    st.stop()

kpis = compute_kpis(filtered_master, filtered_teachers, courses[courses["CourseID"].isin(filtered_master["CourseID"].unique())])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg Teacher Rating", f"{kpis['avg_teacher_rating']:.2f} / 5")
k2.metric("Avg Course Rating", f"{kpis['avg_course_rating']:.2f} / 5")
k3.metric(
    "Rating Consistency Index", f"{kpis['rating_consistency_index']:.2f}",
    help="1 minus the coefficient of variation of an instructor's course ratings, averaged across "
         "instructors with ≥3 delivered courses. Closer to 1 = more consistent quality."
)
k4.metric(
    "Experience Impact Score", f"{kpis['experience_impact_score']:.2f}",
    help="Pearson correlation between YearsOfExperience and TeacherRating across instructors."
)
k5.metric(
    "Enrollment Influence Ratio", f"{kpis['enrollment_influence_ratio']:.1f}×" if not np.isnan(kpis['enrollment_influence_ratio']) else "n/a",
    help="Average enrollments per High-rated (4.0+) instructor divided by average enrollments per "
         "Low-rated (<2.5) instructor. >1 means top-rated instructors attract more enrollments."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs = the brief's analytical modules
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Instructor Leaderboard",
    "📈 Experience vs Rating",
    "🔥 Course Quality Heatmaps",
    "🎯 Expertise Performance",
    "📊 Rating Distributions",
])

# ===========================================================================
# TAB 1 — Instructor Leaderboard
# ===========================================================================
with tab1:
    st.subheader("Instructor Performance Leaderboard")
    st.caption("Ranked by Teacher Rating. Use the sidebar filters to narrow by expertise, experience, or rating range.")

    lb = filtered_summary.sort_values("TeacherRating", ascending=False).reset_index(drop=True)
    lb.index += 1

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Top 10 Instructors**")
        top10 = lb.head(10)[["TeacherName", "Expertise", "YearsOfExperience", "TeacherRating",
                              "AvgCourseRatingDelivered", "Enrollments"]]
        st.dataframe(
            top10.style.format({"TeacherRating": "{:.2f}", "AvgCourseRatingDelivered": "{:.2f}"}),
            width='stretch',
        )
    with c2:
        st.markdown("**Bottom 10 Instructors**")
        bottom10 = lb.tail(10)[["TeacherName", "Expertise", "TeacherRating"]].sort_values("TeacherRating")
        st.dataframe(
            bottom10.style.format({"TeacherRating": "{:.2f}"}),
            width='stretch',
        )

    st.markdown("**Full Leaderboard**")
    fig = px.bar(
        lb.sort_values("TeacherRating", ascending=True),
        x="TeacherRating", y="TeacherName", color="RatingTier",
        color_discrete_map=TIER_COLORS, category_orders={"RatingTier": TIER_ORDER},
        orientation="h", height=max(400, len(lb) * 16),
        labels={"TeacherRating": "Teacher Rating", "TeacherName": ""},
        hover_data={"Expertise": True, "YearsOfExperience": True, "Enrollments": True},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

# ===========================================================================
# TAB 2 — Experience vs Rating
# ===========================================================================
with tab2:
    st.subheader("Does Experience Translate Into Better Ratings?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Years of Experience vs Teacher Rating**")
        corr1 = filtered_teachers["YearsOfExperience"].corr(filtered_teachers["TeacherRating"])
        fig = px.scatter(
            filtered_teachers, x="YearsOfExperience", y="TeacherRating",
            color="Expertise", trendline="ols",
            hover_data=["TeacherName"],
            labels={"YearsOfExperience": "Years of Experience", "TeacherRating": "Teacher Rating"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Pearson correlation: **{corr1:.3f}**" if not pd.isna(corr1) else "Not enough data points for this filter.")

    with col2:
        st.markdown("**Years of Experience vs Course Rating Delivered**")
        exp_course = filtered_master.groupby("TeacherID").agg(
            YearsOfExperience=("YearsOfExperience", "first"),
            AvgCourseRating=("CourseRating", "mean"),
            Expertise=("Expertise", "first"),
            TeacherName=("TeacherName", "first"),
        ).reset_index()
        corr2 = exp_course["YearsOfExperience"].corr(exp_course["AvgCourseRating"])
        fig = px.scatter(
            exp_course, x="YearsOfExperience", y="AvgCourseRating",
            color="Expertise", trendline="ols",
            hover_data=["TeacherName"],
            labels={"YearsOfExperience": "Years of Experience", "AvgCourseRating": "Avg Course Rating Delivered"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Pearson correlation: **{corr2:.3f}**" if not pd.isna(corr2) else "Not enough data points for this filter.")

    st.markdown("---")
    st.markdown("**Experience Bands — looking for diminishing returns / thresholds**")
    bins = [0, 3, 6, 10, 15, 100]
    band_labels = ["0–3 yrs", "4–6 yrs", "7–10 yrs", "11–15 yrs", "16+ yrs"]
    band_df = filtered_teachers.copy()
    band_df["ExperienceBand"] = pd.cut(band_df["YearsOfExperience"], bins=bins, labels=band_labels, right=True)
    band_summary = band_df.groupby("ExperienceBand", observed=True)["TeacherRating"].agg(["mean", "count"]).reset_index()

    fig = px.bar(
        band_summary, x="ExperienceBand", y="mean", text="count",
        labels={"mean": "Avg Teacher Rating", "ExperienceBand": "Experience Band"},
        category_orders={"ExperienceBand": band_labels},
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Bar height = average Teacher Rating within each experience band; label shows sample size (n). "
        "A flattening or declining trend in later bands would indicate diminishing returns to tenure."
    )

# ===========================================================================
# TAB 3 — Course Quality Heatmaps
# ===========================================================================
with tab3:
    st.subheader("Course Quality by Category, Level, and Gender")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Avg Course Rating: Category × Level**")
        pivot1 = filtered_master.pivot_table(
            index="CourseCategory", columns="CourseLevel", values="CourseRating", aggfunc="mean"
        )
        fig = px.imshow(
            pivot1, text_auto=".2f", color_continuous_scale="RdYlGn", aspect="auto",
            labels=dict(color="Avg Rating"),
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("**Avg Course Rating: Instructor Gender × Course Level**")
        pivot2 = filtered_master.pivot_table(
            index="Gender", columns="CourseLevel", values="CourseRating", aggfunc="mean"
        )
        fig = px.imshow(
            pivot2, text_auto=".2f", color_continuous_scale="RdYlGn", aspect="auto",
            labels=dict(color="Avg Rating"),
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.markdown("**Category Ranking — Consistently High vs Low Rated**")
    cat_summary = filtered_master.groupby("CourseCategory")["CourseRating"].agg(
        AvgRating="mean", StdDev="std", Enrollments="count"
    ).reset_index().sort_values("AvgRating", ascending=False)

    fig = px.bar(
        cat_summary, x="AvgRating", y="CourseCategory", orientation="h",
        error_x="StdDev", color="AvgRating", color_continuous_scale="RdYlGn",
        labels={"AvgRating": "Avg Course Rating", "CourseCategory": ""},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch')
    st.caption("Error bars show rating standard deviation within each category — wider bars mean less consistent quality.")

# ===========================================================================
# TAB 4 — Expertise Performance
# ===========================================================================
with tab4:
    st.subheader("Expertise-Based Performance Insights")

    expertise_summary = filtered_master.groupby("Expertise").agg(
        AvgTeacherRating=("TeacherRating", "mean"),
        AvgCourseRating=("CourseRating", "mean"),
        Enrollments=("TransactionID", "count"),
        InstructorCount=("TeacherID", "nunique"),
    ).reset_index().sort_values("AvgCourseRating", ascending=False)

    fig = px.bar(
        expertise_summary, x="Expertise", y=["AvgTeacherRating", "AvgCourseRating"],
        barmode="group", labels={"value": "Average Rating", "Expertise": "", "variable": "Metric"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-30)
    st.plotly_chart(fig, width='stretch')

    st.markdown("**Expertise Summary Table**")
    st.dataframe(
        expertise_summary.style.format({"AvgTeacherRating": "{:.2f}", "AvgCourseRating": "{:.2f}"})
        width='stretch',
    )

    gap = expertise_summary.copy()
    gap["Gap (Teacher − Course)"] = gap["AvgTeacherRating"] - gap["AvgCourseRating"]
    st.markdown("**Where teacher reputation and actual course outcomes diverge**")
    fig = px.bar(
        gap.sort_values("Gap (Teacher − Course)"), x="Gap (Teacher − Course)", y="Expertise",
        orientation="h", color="Gap (Teacher − Course)", color_continuous_scale="RdBu_r",
        labels={"Gap (Teacher − Course)": "Teacher Rating − Course Rating"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Positive bars: instructors in this expertise are rated higher than the courses they deliver "
        "(possible training/content gap). Negative bars: courses outperform instructor reputation."
    )

# ===========================================================================
# TAB 5 — Rating Distributions & Instructor Impact Tiers
# ===========================================================================
with tab5:
    st.subheader("Rating Distributions & Instructor Impact on Course Success")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribution of Teacher Ratings**")
        fig = px.histogram(filtered_teachers, x="TeacherRating", nbins=20, color_discrete_sequence=["#4C72B0"])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), bargap=0.05)
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("**Distribution of Course Ratings**")
        filt_courses = courses[courses["CourseID"].isin(filtered_master["CourseID"].unique())]
        fig = px.histogram(filt_courses, x="CourseRating", nbins=20, color_discrete_sequence=["#55A868"])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), bargap=0.05)
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.markdown("**Course Ratings by Instructor Rating Tier**")
    tier_master = filtered_master.copy()
    tier_master["RatingTier"] = tier_master["TeacherRating"].apply(
        lambda r: "High (4.0+)" if r >= 4.0 else ("Mid (2.5–3.99)" if r >= 2.5 else "Low (<2.5)")
    )

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(
            tier_master, x="RatingTier", y="CourseRating", color="RatingTier",
            color_discrete_map=TIER_COLORS, category_orders={"RatingTier": TIER_ORDER},
            labels={"CourseRating": "Course Rating", "RatingTier": "Instructor Tier"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.caption("Course-rating spread delivered by instructors in each tier.")

    with col4:
        enroll_tier = tier_master.groupby(["TeacherID", "RatingTier"]).size().reset_index(name="Enrollments")
        enroll_tier_avg = enroll_tier.groupby("RatingTier")["Enrollments"].mean().reset_index()
        fig = px.bar(
            enroll_tier_avg, x="RatingTier", y="Enrollments", color="RatingTier",
            color_discrete_map=TIER_COLORS, category_orders={"RatingTier": TIER_ORDER},
            labels={"Enrollments": "Avg Enrollments per Instructor"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.caption("Average enrollment volume per instructor, grouped by their rating tier.")

st.markdown("---")
st.caption(
    "EduPro Instructor Performance & Course Quality Evaluation · "
    f"{len(teachers)} instructors · {len(courses)} courses · {len(transactions):,} enrollment transactions"
)