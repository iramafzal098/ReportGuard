"""ReportGuard data-quality and issue-detection engine.

Lightweight, deterministic, pandas-only engine. It never mutates the input DataFrame.
Stable interface for downstream UI / explanation layers:
    analyze_data(df, reference_sets=None) -> dict
    detect_issues(df, reference_sets=None) -> list[dict]
    cluster_issues(issues) -> list[dict]
    calculate_issue_severity(issue_type, affected_rows, total_rows, **kwargs) -> str
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

MANDATORY_FIELDS = ["ReportID", "SalesPersonID", "ProductID", "Quantity", "TotalSalesValue"]
NUMERIC_FIELDS = ["Quantity", "TotalSalesValue"]
ID_PATTERNS = {
    "ReportID": re.compile(r"^Rep\d+$", re.IGNORECASE),
    "SalesPersonID": re.compile(r"^C\d+$", re.IGNORECASE),
    "ProductID": re.compile(r"^PR\d+$", re.IGNORECASE),
}

SEVERITY_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}

RULE_META = {
    "missing_mandatory_fields": ("Completeness Failure", "Data ingestion / field population issue"),
    "duplicate_report_id": ("Report ID Duplication", "Duplicate submission or report-generation issue"),
    "invalid_numeric_values": ("Numeric Validation Failure", "Malformed or non-positive transaction values"),
    "extreme_values": ("Extreme Transaction Values", "Unusual transaction magnitude or entry error"),
    "potential_value_inconsistency": ("Potential Value Inconsistency", "Pricing, unit-value, or calculation inconsistency"),
    "invalid_id_mapping": ("ID Mapping Failure", "Master-data or identifier mapping inconsistency"),
    "conflicting_records": ("Conflicting Report Records", "Repeated report identifier with inconsistent attributes"),
    "suspicious_combination": ("Unusual Transaction Combination", "Rare or atypical salesperson-product-value combination"),
}


def _clean_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _row_ids(df: pd.DataFrame, mask: pd.Series) -> list[str]:
    if "ReportID" not in df.columns:
        return [str(i) for i in df.index[mask]]
    return df.loc[mask, "ReportID"].astype("string").fillna("<missing>").astype(str).tolist()


def _issue(issue_type: str, mask: pd.Series, df: pd.DataFrame, evidence: str,
           description: str, rule_triggered: str, confidence: float,
           root_cause: str, severity: Optional[str] = None,
           extra: Optional[dict] = None) -> dict:
    affected = int(mask.sum())
    sev = severity or calculate_issue_severity(issue_type, affected, len(df), mandatory=(issue_type == "missing_mandatory_fields"))
    out = {
        "issue_id": f"ISSUE-{issue_type.upper()}-001",
        "issue_type": issue_type,
        "severity": sev,
        "affected_rows": affected,
        "affected_record_ids": _row_ids(df, mask)[:1000],
        "evidence": evidence,
        "description": description,
        "rule_triggered": rule_triggered,
        "confidence": round(float(confidence), 3),
        "possible_root_cause_category": root_cause,
    }
    if extra:
        out.update(extra)
    return out


def calculate_issue_severity(issue_type: str, affected_rows: int, total_rows: int,
                              mandatory: bool = False, materially_changes_totals: bool = False,
                              conflict: bool = False, invalid_mapping: bool = False,
                              **_: Any) -> str:
    """Deterministically score severity from impact, not randomly.

    Critical is reserved for widespread integrity/completeness failures or issues
    that directly invalidate report totals. High is used for meaningful mapping,
    duplicate, numeric, or conflict failures. Medium/Low cover narrower anomalies.
    """
    n = max(int(total_rows), 1)
    rate = affected_rows / n
    if mandatory and rate >= 0.10:
        return "Critical"
    if materially_changes_totals and rate >= 0.02:
        return "Critical"
    if conflict and affected_rows >= max(10, int(n * 0.01)):
        return "Critical"
    if invalid_mapping and affected_rows >= max(25, int(n * 0.005)):
        return "High"
    if issue_type in {"duplicate_report_id", "conflicting_records"} and affected_rows >= max(10, int(n * 0.002)):
        return "High"
    if issue_type == "missing_mandatory_fields":
        return "High" if affected_rows >= 5 else "Medium"
    if issue_type == "invalid_numeric_values":
        return "High" if affected_rows >= max(10, int(n * 0.005)) else "Medium"
    if issue_type == "extreme_values":
        return "High" if affected_rows >= max(50, int(n * 0.01)) else "Medium"
    if issue_type == "potential_value_inconsistency":
        return "High" if affected_rows >= max(50, int(n * 0.02)) else "Medium"
    if issue_type == "suspicious_combination":
        return "Medium" if affected_rows >= 5 else "Low"
    return "Medium" if rate >= 0.001 else "Low"


def _empty_issue(issue_type: str) -> Optional[dict]:
    return None


def _detect_missing(df: pd.DataFrame) -> list[dict]:
    findings = []
    for c in MANDATORY_FIELDS:
        if c not in df.columns:
            mask = pd.Series(True, index=df.index)
        else:
            s = df[c]
            mask = s.isna() | _clean_str(s).eq("")
        if mask.any():
            findings.append(_issue(
                "missing_mandatory_fields", mask, df,
                f"{int(mask.sum())} rows have missing {c}.",
                f"Mandatory field {c} is empty or absent.",
                f"{c} is null, blank, or missing from the schema.", 0.99,
                "Data completeness / ingestion",
                extra={"field": c, "cluster_name": f"Missing {c}"},
            ))
    return findings


def _detect_duplicates(df: pd.DataFrame) -> Optional[dict]:
    if "ReportID" not in df.columns:
        return None
    ids = _clean_str(df["ReportID"])
    mask = ids.duplicated(keep=False) & ids.ne("") & ids.notna()
    if not mask.any():
        return None
    dup_groups = int(ids[mask].nunique())
    return _issue(
        "duplicate_report_id", mask, df,
        f"{dup_groups} ReportID values occur more than once; {int(mask.sum())} rows are involved.",
        "Repeated ReportIDs can double-count or overwrite reporting records.",
        "ReportID appears in multiple rows.", 0.99, "Duplicate submission / report generation",
        extra={"duplicate_report_id_groups": dup_groups},
    )


def _detect_numeric(df: pd.DataFrame) -> list[dict]:
    findings = []
    for c in NUMERIC_FIELDS:
        if c not in df.columns:
            continue
        raw = df[c]
        numeric = pd.to_numeric(raw, errors="coerce")
        non_numeric = numeric.isna() & raw.notna() & _clean_str(raw).ne("")
        invalid = non_numeric.copy()
        if c == "Quantity":
            invalid |= numeric.le(0)
        else:
            invalid |= numeric.lt(0)
        if invalid.any():
            evidence = {"non_numeric": int(non_numeric.sum()), "invalid_range": int((invalid & ~non_numeric).sum())}
            findings.append(_issue(
                "invalid_numeric_values", invalid, df,
                f"{c}: {evidence}",
                f"{c} contains malformed or out-of-range numeric values.",
                "Non-numeric value or invalid numeric range.", 0.995,
                "Data validation / manual entry",
                extra={"numeric_evidence": evidence, "field": c, "cluster_name": f"Invalid {c}"},
            ))
    return findings


def _iqr_mask(s: pd.Series) -> tuple[pd.Series, dict]:
    x = pd.to_numeric(s, errors="coerce")
    valid = x.dropna()
    if len(valid) < 8:
        return pd.Series(False, index=s.index), {}
    q1, q3 = valid.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr <= 0:
        return pd.Series(False, index=s.index), {"q1": float(q1), "q3": float(q3)}
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = x.lt(low) | x.gt(high)
    return mask.fillna(False), {"q1": float(q1), "q3": float(q3), "lower": float(low), "upper": float(high)}


def _detect_extremes(df: pd.DataFrame) -> list[dict]:
    findings = []
    for c in NUMERIC_FIELDS:
        if c not in df.columns:
            continue
        mask, stats = _iqr_mask(df[c])
        if mask.any():
            findings.append(_issue(
                "extreme_values", mask, df,
                f"IQR outlier thresholds for {c}: {stats}",
                f"{c} contains values well outside its normal interquartile range.",
                "1.5 × IQR rule.", 0.90,
                "Outlier transaction / possible entry error",
                extra={"outlier_statistics": stats, "field": c, "cluster_name": f"Extreme {c} Values"},
            ))
    return findings


def _detect_value_consistency(df: pd.DataFrame) -> Optional[dict]:
    if not {"ProductID", "Quantity", "TotalSalesValue"}.issubset(df.columns):
        return None
    q = pd.to_numeric(df["Quantity"], errors="coerce")
    v = pd.to_numeric(df["TotalSalesValue"], errors="coerce")
    valid = q.gt(0) & q.notna() & v.notna() & v.ge(0)
    if valid.sum() < 20:
        return None
    work = pd.DataFrame({"ProductID": _clean_str(df["ProductID"]), "unit": v / q})
    work = work.where(valid, np.nan)
    stats = work.groupby("ProductID")["unit"].agg(["median", "count"])
    stats = stats[stats["count"] >= 5]
    med = df["ProductID"].map(stats["median"])
    cnt = df["ProductID"].map(stats["count"])
    ratio = (v / q) / med
    mask = valid & cnt.ge(5) & ((ratio.gt(3)) | (ratio.lt(1/3)))
    mask = mask.fillna(False)
    if not mask.any():
        return None
    return _issue(
        "potential_value_inconsistency", mask, df,
        "For products with >=5 observations, unit value (TotalSalesValue / Quantity) differs by more than 3x from that product's median.",
        "Transaction value is materially inconsistent with the usual unit-value pattern for the same product. This is a potential inconsistency, not proof of an incorrect formula.",
        "Product-level median unit value comparison; flag when ratio is >3x or <1/3x.", 0.82,
        "Pricing / calculation / transaction-entry inconsistency",
    )


def _detect_ids(df: pd.DataFrame, reference_sets: Optional[dict]) -> list[dict]:
    findings = []
    for c, pat in ID_PATTERNS.items():
        if c not in df.columns or c == "ReportID":
            continue
        s = _clean_str(df[c])
        malformed = s.isna() | s.eq("") | ~s.str.match(pat)
        ref = (reference_sets or {}).get(c)
        unseen = pd.Series(False, index=df.index)
        if ref is not None:
            unseen = ~s.isin(set(map(str, ref))) & ~s.isna() & s.ne("")
        mask = malformed | unseen
        if mask.any():
            evidence = {"malformed": int(malformed.sum()), "unseen": int(unseen.sum())}
            findings.append(_issue(
                "invalid_id_mapping", mask, df,
                f"{c}: {evidence}",
                f"{c} is malformed or not present in the trusted reference set.",
                "Regex structure and reference-set membership check.", 0.98,
                "Master-data / identifier mapping",
                extra={"id_evidence": evidence, "field": c, "cluster_name": f"{c} Mapping Failure"},
            ))
    return findings


def _detect_conflicts(df: pd.DataFrame) -> Optional[dict]:
    if "ReportID" not in df.columns:
        return None
    ids = _clean_str(df["ReportID"])
    dup = ids[ids.duplicated(keep=False) & ids.notna() & ids.ne("")]
    if dup.empty: return None
    compare_cols = [c for c in MANDATORY_FIELDS if c != "ReportID" and c in df.columns]
    conflict_ids = []
    for rid, g in df.loc[dup.index].groupby("ReportID", dropna=False):
        if len(g) > 1 and any(g[c].astype("string").nunique(dropna=False) > 1 for c in compare_cols):
            conflict_ids.append(rid)
    if not conflict_ids: return None
    mask = ids.isin(conflict_ids)
    return _issue(
        "conflicting_records", mask, df,
        f"{len(conflict_ids)} duplicated ReportID groups contain differing field values.",
        "The same ReportID appears with conflicting transaction attributes, so the reporting record cannot be treated as a single consistent event.",
        "Repeated ReportID has at least one differing mandatory attribute.", 0.99,
        "Duplicate submission / conflicting update",
        extra={"conflicting_report_ids": [str(x) for x in conflict_ids[:1000]]},
    )


def _detect_combinations(df: pd.DataFrame) -> Optional[dict]:
    required = {"SalesPersonID", "ProductID", "Quantity", "TotalSalesValue"}
    if not required.issubset(df.columns): return None
    q = pd.to_numeric(df["Quantity"], errors="coerce")
    v = pd.to_numeric(df["TotalSalesValue"], errors="coerce")
    valid = q.gt(0) & v.ge(0) & q.notna() & v.notna()
    pair = df["SalesPersonID"].astype("string") + "|" + df["ProductID"].astype("string")
    freq = pair.value_counts()
    rare = pair.map(freq).le(1)
    product_q95 = q.where(valid).groupby(df["ProductID"]).transform(lambda x: x.quantile(.95) if len(x.dropna()) >= 5 else np.nan)
    mask = valid & rare & product_q95.notna() & q.gt(product_q95 * 2)
    if not mask.any(): return None
    return _issue(
        "suspicious_combination", mask, df,
        "Flags valid SalesPersonID/ProductID pairs observed once where Quantity is >2x that product's 95th-percentile quantity.",
        "A rare salesperson-product combination also carries an unusually large quantity; it is an investigation candidate.",
        "Rare valid mapping + quantity above 2x product 95th percentile.", 0.76,
        "Unusual transaction pattern",
    )


def detect_issues(df: pd.DataFrame, reference_sets: Optional[dict] = None) -> list[dict]:
    """Run all supported checks and return aggregated rule findings."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    findings = []
    for detector in [_detect_missing, _detect_duplicates, _detect_numeric, _detect_extremes,
                     _detect_value_consistency, _detect_ids, _detect_conflicts, _detect_combinations]:
        result = detector(df, reference_sets) if detector is _detect_ids else detector(df)
        if result is None:
            continue
        if isinstance(result, dict):
            result = [result]
        findings.extend(x for x in result if x and x["affected_rows"])
    for i, item in enumerate(findings, 1):
        item["issue_id"] = f"ISSUE-{i:03d}"
    return findings


def cluster_issues(issues: Iterable[dict]) -> list[dict]:
    """Aggregate rule findings into root-cause-oriented issue groups."""
    groups: dict[str, dict] = {}
    for issue in issues:
        key = issue.get("cluster_name") or RULE_META.get(issue["issue_type"], (issue["issue_type"], issue.get("possible_root_cause_category", "Other")))[0]
        if key not in groups:
            groups[key] = {
                "cluster_id": f"CLUSTER-{len(groups)+1:03d}",
                "cluster_name": key,
                "severity": issue["severity"],
                "issue_types": [],
                "affected_rows": 0,
                "affected_record_ids": [],
                "evidence": [],
                "likely_root_cause": RULE_META.get(issue["issue_type"], (key, "Other"))[1],
                "confidence": issue["confidence"],
                "recommended_action": _recommend(issue["issue_type"]),
            }
        g = groups[key]
        g["issue_types"].append(issue["issue_type"])
        g["affected_rows"] += issue["affected_rows"]
        g["affected_record_ids"] = list(dict.fromkeys(g["affected_record_ids"] + issue.get("affected_record_ids", [])))[:1000]
        g["evidence"].append(issue["evidence"])
        if SEVERITY_ORDER.get(issue["severity"], 0) > SEVERITY_ORDER.get(g["severity"], 0):
            g["severity"] = issue["severity"]
        g["confidence"] = round(max(g["confidence"], issue["confidence"]), 3)
    result = list(groups.values())
    result.sort(key=lambda x: (-SEVERITY_ORDER.get(x["severity"], 0), -x["affected_rows"], x["cluster_name"]))
    for i, g in enumerate(result, 1): g["cluster_id"] = f"CLUSTER-{i:03d}"
    return result


def _recommend(issue_type: str) -> str:
    return {
        "missing_mandatory_fields": "Trace the source extract/form and populate mandatory fields before publishing.",
        "duplicate_report_id": "Check report generation and deduplicate or reconcile repeated ReportIDs.",
        "invalid_numeric_values": "Validate source entry rules and correct malformed or non-positive transaction values.",
        "extreme_values": "Review flagged transactions against source documents; do not auto-delete outliers.",
        "potential_value_inconsistency": "Compare with product pricing/master data and verify the transaction calculation.",
        "invalid_id_mapping": "Reconcile identifiers against the current salesperson/product master mappings.",
        "conflicting_records": "Reconcile repeated ReportIDs and determine the authoritative version.",
        "suspicious_combination": "Investigate the salesperson-product transaction against source evidence and business context.",
    }.get(issue_type, "Investigate the affected records and source process.")


def analyze_data(df: pd.DataFrame, reference_sets: Optional[dict] = None) -> dict:
    """Primary stable API consumed by Person 2 and Person 3."""
    issues = detect_issues(df, reference_sets=reference_sets)
    clusters = cluster_issues(issues)
    severity = Counter(i["severity"] for i in issues)
    raw_issue_count = int(sum(i["affected_rows"] for i in issues))
    return {
        "summary": {
            "total_rows": int(len(df)),
            "total_issues": len(issues),
            "raw_issue_count": raw_issue_count,
            "issue_groups": len(clusters),
            "critical": int(severity.get("Critical", 0)),
            "high": int(severity.get("High", 0)),
            "medium": int(severity.get("Medium", 0)),
            "low": int(severity.get("Low", 0)),
        },
        "issues": issues,
        "clusters": clusters,
    }


def build_reference_sets(df: pd.DataFrame) -> dict:
    """Build valid ID sets from a trusted baseline/clean dataset."""
    return {c: set(_clean_str(df[c]).dropna().tolist()) for c in ["ReportID", "SalesPersonID", "ProductID"] if c in df.columns}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ReportGuard quality engine")
    parser.add_argument("input", help="Input .xlsx/.csv")
    args = parser.parse_args()
    df = pd.read_excel(args.input) if args.input.lower().endswith((".xlsx", ".xls")) else pd.read_csv(args.input)
    result = analyze_data(df)
    print("Records:", result["summary"]["total_rows"])
    print("Detected issue groups:", result["summary"]["total_issues"])
    print("Raw affected issue instances:", result["summary"]["raw_issue_count"])
    print("Severity:", {k: result["summary"][k.lower()] for k in ["Critical", "High", "Medium", "Low"]})
    print("Top 5:")
    for x in sorted(result["issues"], key=lambda z: (-SEVERITY_ORDER[z["severity"]], -z["affected_rows"]))[:5]:
        print(f"  {x['severity']:<8} {x['issue_type']:<32} {x['affected_rows']:>6} rows")
