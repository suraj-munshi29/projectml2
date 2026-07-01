"""
EduPro - Instructor Performance & Course Quality Evaluation
Data pipeline: loading, joining, validation, and KPI computation.

Data model notes (validated against actual files):
- Teachers (60 rows), Courses (60 rows), Transactions (10,000 rows), Users (3,000 rows).
- No nulls, no duplicate IDs, all foreign keys resolve cleanly.
- IMPORTANT: TeacherID <-> CourseID is a genuine MANY-TO-MANY relationship.
  Each course is delivered by ~15 different teachers (different enrollment instances),
  and each teacher delivers ~15 different courses. There is no fixed 1:1 "course owner".
  Therefore the correct unit of analysis is the TRANSACTION (one enrollment = one
  teacher delivering one course to one user), and all instructor/course aggregates
  are built by joining Teachers <- Transactions -> Courses on TeacherID and CourseID.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Loading & joining
# ---------------------------------------------------------------------------

def load_raw():
    teachers = pd.read_csv(DATA_DIR / "EduPro_Online_Platform_xlsx_-_Teachers.csv")
    courses = pd.read_csv(DATA_DIR / "EduPro_Online_Platform_xlsx_-_Courses.csv")
    transactions = pd.read_csv(DATA_DIR / "EduPro_Online_Platform_xlsx_-_Transactions.csv")
    users = pd.read_csv(DATA_DIR / "EduPro_Online_Platform_xlsx_-_Users.csv")

    transactions["TransactionDate"] = pd.to_datetime(
        transactions["TransactionDate"], format="%d/%m/%Y", errors="coerce"
    )
    return teachers, courses, transactions, users


def validate(teachers, courses, transactions, users):
    """Returns a list of human-readable validation findings."""
    findings = []

    for name, df, key in [
        ("Teachers", teachers, "TeacherID"),
        ("Courses", courses, "CourseID"),
        ("Transactions", transactions, "TransactionID"),
        ("Users", users, "UserID"),
    ]:
        nulls = int(df.isnull().sum().sum())
        dupes = int(df[key].duplicated().sum())
        findings.append(f"{name}: {len(df)} rows, {nulls} null cells, {dupes} duplicate {key}")

    orphan_teachers = set(transactions.TeacherID) - set(teachers.TeacherID)
    orphan_courses = set(transactions.CourseID) - set(courses.CourseID)
    orphan_users = set(transactions.UserID) - set(users.UserID)
    findings.append(f"Orphan TeacherIDs in Transactions: {len(orphan_teachers)}")
    findings.append(f"Orphan CourseIDs in Transactions: {len(orphan_courses)}")
    findings.append(f"Orphan UserIDs in Transactions: {len(orphan_users)}")

    avg_teachers_per_course = transactions.groupby("CourseID")["TeacherID"].nunique().mean()
    avg_courses_per_teacher = transactions.groupby("TeacherID")["CourseID"].nunique().mean()
    findings.append(
        f"Avg distinct teachers per course: {avg_teachers_per_course:.1f} "
        f"(confirms many-to-many; analysis is done at the transaction level)"
    )
    findings.append(f"Avg distinct courses per teacher: {avg_courses_per_teacher:.1f}")

    return findings


def build_master(teachers, courses, transactions):
    """
    The single source of truth for all analysis: one row per enrollment
    (transaction), enriched with the delivering teacher's profile and the
    course's attributes.
    """
    master = (
        transactions
        .merge(teachers, on="TeacherID", how="left")
        .merge(courses, on="CourseID", how="left")
    )
    return master


# ---------------------------------------------------------------------------
# Instructor profile aggregates
# ---------------------------------------------------------------------------

def instructor_summary(master: pd.DataFrame, teachers: pd.DataFrame) -> pd.DataFrame:
    """
    One row per teacher: their static profile fields plus performance
    measured across every course they actually delivered (via transactions).
    """
    agg = master.groupby("TeacherID").agg(
        Enrollments=("TransactionID", "count"),
        DistinctCoursesTaught=("CourseID", "nunique"),
        AvgCourseRatingDelivered=("CourseRating", "mean"),
        TotalRevenue=("Amount", "sum"),
    ).reset_index()

    summary = teachers.merge(agg, on="TeacherID", how="left")
    summary[["Enrollments", "DistinctCoursesTaught", "TotalRevenue"]] = (
        summary[["Enrollments", "DistinctCoursesTaught", "TotalRevenue"]].fillna(0)
    )
    return summary


def rating_tier(rating: float) -> str:
    if rating >= 4.0:
        return "High (4.0+)"
    elif rating >= 2.5:
        return "Mid (2.5–3.99)"
    else:
        return "Low (<2.5)"


def add_rating_tiers(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["RatingTier"] = summary["TeacherRating"].apply(rating_tier)
    return summary


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def compute_kpis(master: pd.DataFrame, teachers: pd.DataFrame, courses: pd.DataFrame) -> dict:
    avg_teacher_rating = teachers["TeacherRating"].mean()
    avg_course_rating = courses["CourseRating"].mean()

    # Rating Consistency Index: 1 - (std / mean) on a 0-1 scale, higher = more consistent.
    # Computed on teachers with at least 3 delivered courses to be meaningful.
    course_rating_std_per_teacher = master.groupby("TeacherID")["CourseRating"].agg(["std", "mean", "count"])
    consistent_subset = course_rating_std_per_teacher[course_rating_std_per_teacher["count"] >= 3]
    rating_consistency_index = (
        1 - (consistent_subset["std"] / consistent_subset["mean"]).clip(0, 1)
    ).mean()

    # Experience Impact Score: correlation between YearsOfExperience and TeacherRating
    experience_impact_score = teachers["YearsOfExperience"].corr(teachers["TeacherRating"])

    # Enrollment Influence Ratio: avg enrollments-per-course for High-rated vs Low-rated instructors
    summ = add_rating_tiers(instructor_summary(master, teachers))
    enroll_by_tier = summ.groupby("RatingTier")["Enrollments"].mean()
    high = enroll_by_tier.get("High (4.0+)", np.nan)
    low = enroll_by_tier.get("Low (<2.5)", np.nan)
    enrollment_influence_ratio = (high / low) if (low and low > 0) else np.nan

    return {
        "avg_teacher_rating": avg_teacher_rating,
        "avg_course_rating": avg_course_rating,
        "rating_consistency_index": rating_consistency_index,
        "experience_impact_score": experience_impact_score,
        "enrollment_influence_ratio": enrollment_influence_ratio,
    }


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    teachers, courses, transactions, users = load_raw()

    print("=== VALIDATION ===")
    for line in validate(teachers, courses, transactions, users):
        print(" -", line)

    master = build_master(teachers, courses, transactions)
    print("\nMaster table shape:", master.shape)
    assert master.isnull().sum().sum() == 0, "Unexpected nulls after join!"

    summ = add_rating_tiers(instructor_summary(master, teachers))
    print("\nInstructor summary sample:")
    print(summ.head())
    print("\nRating tier counts:")
    print(summ["RatingTier"].value_counts())

    kpis = compute_kpis(master, teachers, courses)
    print("\n=== KPIs ===")
    for k, v in kpis.items():
        print(f" - {k}: {v:.3f}" if isinstance(v, (int, float)) and not pd.isna(v) else f" - {k}: {v}")