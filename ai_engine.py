"""
ReportGuard - Person 2: AI Intelligence Layer

Consumes the deterministic output from Person 1's quality_engine.py.

Pipeline:
Person 1 detection -> prioritize -> investigate -> project impact -> audit
"""

import json
import os
from typing import Any, Dict, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _severity_score(severity: str) -> int:
    """Convert severity into a numeric priority score."""
    values = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }
    return values.get(str(severity).lower(), 0)


def _confidence_label(confidence: Any) -> str:
    """Convert Person 1's numeric confidence into a readable label."""
    if isinstance(confidence, (int, float)):
        if confidence >= 0.90:
            return "High"
        elif confidence >= 0.70:
            return "Medium"
        return "Low"

    return str(confidence) if confidence else "Unknown"


def _affected_rows(issue: Dict[str, Any]) -> int:
    """Read Person 1's affected_rows field safely."""
    value = issue.get("affected_rows", 0)

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _total_rows(results_or_summary: Dict[str, Any]) -> int:
    """Read total_rows from either full results or summary."""
    if "summary" in results_or_summary:
        summary = results_or_summary.get("summary", {})
    else:
        summary = results_or_summary

    try:
        return int(summary.get("total_rows", 0))
    except (TypeError, ValueError):
        return 0


def _safe_json(value: Any) -> Any:
    """Return a JSON-safe version of a value."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


# ============================================================
# 1. READINESS
# ============================================================

def calculate_readiness(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate a deterministic report-readiness score.

    This uses Person 1's summary.
    The LLM is NOT used for this score.
    """

    summary = results.get("summary", {})

    critical = int(summary.get("critical", 0) or 0)
    high = int(summary.get("high", 0) or 0)
    medium = int(summary.get("medium", 0) or 0)
    low = int(summary.get("low", 0) or 0)

    total_issues = int(summary.get("total_issues", 0) or 0)

    # Penalty-based deterministic score.
    penalty = (
        critical * 20
        + high * 8
        + medium * 3
        + low * 1
    )

    score = max(0, min(100, 100 - penalty))

    if score >= 80:
        status = "ready"
    elif score >= 60:
        status = "needs_review"
    elif score >= 40:
        status = "at_risk"
    else:
        status = "not_ready"

    return {
        "score": score,
        "status": status,
        "critical_issues": critical,
        "high_issues": high,
        "medium_issues": medium,
        "low_issues": low,
        "total_issues": total_issues,
    }


# ============================================================
# 2. PRIORITIZE ISSUES
# ============================================================

def prioritize_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank Person 1's issues by severity, affected scope,
    and detection confidence.
    """

    ranked = []

    for issue in issues:
        severity = issue.get("severity", "Low")
        affected = _affected_rows(issue)

        confidence = issue.get("confidence", 0)

        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0

        severity_value = _severity_score(severity)

        # Scope is capped so a huge affected-row count
        # does not completely dominate severity.
        scope_score = min(affected / 1000, 10)

        priority_score = (
            severity_value * 20
            + scope_score * 5
            + confidence_value * 5
        )

        ranked.append({
            "rank": 0,
            "issue_id": issue.get("issue_id"),
            "issue_type": issue.get("issue_type"),
            "name": issue.get(
                "cluster_name",
                issue.get("issue_type", "Unknown issue")
            ),
            "severity": severity,
            "affected_rows": affected,
            "confidence": _confidence_label(confidence),
            "priority_score": round(priority_score, 2),
            "original_issue": issue,
        })

    ranked.sort(
        key=lambda x: x["priority_score"],
        reverse=True
    )

    for index, item in enumerate(ranked, start=1):
        item["rank"] = index

    return ranked


# ============================================================
# 3. AI INVESTIGATION
# ============================================================

def _fallback_investigation(
    issue: Dict[str, Any],
    dataset_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deterministic fallback used when the LLM is unavailable.
    """

    affected = _affected_rows(issue)
    total = _total_rows(dataset_summary)

    if total > 0:
        percentage = round((affected / total) * 100, 2)
    else:
        percentage = None

    root_cause = issue.get(
        "possible_root_cause_category",
        "Not specified by detection engine"
    )

    recommendation = issue.get(
        "recommended_action",
        "Review the affected records and validate the upstream data source."
    )

    return {
        "issue_summary": (
            f"{affected} records were flagged by the deterministic "
            f"quality engine."
        ),

        "affected_scope": {
            "rows": affected,
            "percentage_of_dataset": percentage,
        },

        "likely_root_cause": (
            f"Possible category: {root_cause}. "
            "This is an interpretation, not a confirmed root cause."
        ),

        "interpretation": issue.get(
            "description",
            "The quality engine identified a data-quality condition."
        ),

        "recommendation": recommendation,

        "confidence": _confidence_label(
            issue.get("confidence")
        ),

        "assumptions": [
            "The deterministic rule correctly identified the affected records.",
            "The possible root-cause category is a hypothesis rather than proof."
        ],

        "source": "deterministic_fallback",
    }


def investigate_issue(
    issue: Dict[str, Any],
    dataset_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use an LLM to investigate one issue.

    The full CSV is NEVER sent to the LLM.
    Only compact issue evidence and aggregate information are sent.
    """

    fallback = _fallback_investigation(
        issue,
        dataset_summary
    )

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or OpenAI is None:
        return fallback

    model = os.getenv(
        "OPENAI_MODEL",
        "gpt-6-luna"
    )

    affected = _affected_rows(issue)
    total = _total_rows(dataset_summary)

    percentage = (
        round((affected / total) * 100, 2)
        if total > 0
        else None
    )

    compact_context = {
        "dataset_summary": {
            "total_rows": total,
            "total_issues": dataset_summary.get("total_issues"),
            "issue_groups": dataset_summary.get("issue_groups"),
            "critical": dataset_summary.get("critical"),
            "high": dataset_summary.get("high"),
            "medium": dataset_summary.get("medium"),
            "low": dataset_summary.get("low"),
        },

        "issue": {
            "issue_id": issue.get("issue_id"),
            "issue_type": issue.get("issue_type"),
            "cluster_name": issue.get("cluster_name"),
            "severity": issue.get("severity"),
            "affected_rows": affected,
            "affected_percentage": percentage,
            "evidence": issue.get("evidence"),
            "description": issue.get("description"),
            "rule_triggered": issue.get("rule_triggered"),
            "confidence": issue.get("confidence"),
            "possible_root_cause_category": issue.get(
                "possible_root_cause_category"
            ),
            "recommended_action": issue.get(
                "recommended_action"
            ),
        }
    }

    prompt = f"""
You are the investigation layer of ReportGuard.

A deterministic data-quality engine has already detected an issue.
Your job is to INTERPRET the evidence, not replace the detection engine.

IMPORTANT:
- Do not invent facts.
- Do not claim a root cause is confirmed.
- Clearly distinguish deterministic facts from AI interpretation.
- If evidence is insufficient, say so.
- Projected outcomes are estimates, not guarantees.
- Return ONLY valid JSON.

Required JSON structure:

{{
  "issue_summary": "short explanation",
  "likely_root_cause": "possible explanation, clearly labeled as a hypothesis",
  "interpretation": "what the evidence may indicate",
  "recommendation": "practical next action",
  "confidence": "High/Medium/Low",
  "assumptions": ["..."],
  "source": "llm"
}}

Here is the compact evidence:

{json.dumps(compact_context, indent=2)}
"""

    try:
        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model=model,
            input=prompt
        )

        text = response.output_text.strip()

        parsed = json.loads(text)

        # Ensure required fields exist.
        required = [
            "issue_summary",
            "likely_root_cause",
            "interpretation",
            "recommendation",
            "confidence",
            "assumptions",
        ]

        for field in required:
            if field not in parsed:
                return fallback

        parsed["source"] = "llm"

        return parsed

    except Exception:
        # The application should still work if the API fails.
        return fallback


# ============================================================
# 4. PROJECT IMPACT
# ============================================================

def project_impact(
    issue: Dict[str, Any],
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Estimate the possible readiness improvement if one issue
    were resolved.

    This is explicitly a PROJECTED impact, not a guarantee.
    """

    current = calculate_readiness(results)
    current_score = current["score"]

    severity = str(
        issue.get("severity", "Low")
    ).lower()

    affected = _affected_rows(issue)
    total = _total_rows(results)

    if total > 0:
        affected_ratio = affected / total
    else:
        affected_ratio = 0

    severity_gain = {
        "critical": 25,
        "high": 15,
        "medium": 8,
        "low": 3,
    }.get(severity, 2)

    scope_adjustment = min(
        int(affected_ratio * 10),
        10
    )

    improvement = min(
        severity_gain + scope_adjustment,
        30
    )

    projected_score = min(
        100,
        current_score + improvement
    )

    return {
        "label": "Projected impact",
        "current_readiness": current_score,
        "projected_readiness": projected_score,
        "projected_improvement": (
            projected_score - current_score
        ),
        "issue_id": issue.get("issue_id"),
        "note": (
            "This is an estimate based on issue severity "
            "and affected scope, not a guaranteed outcome."
        ),
    }


# ============================================================
# 5. AUDIT ENTRY
# ============================================================

def generate_audit_entry(
    issue: Dict[str, Any],
    investigation: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine deterministic facts from Person 1 with
    the investigation from Person 2.
    """

    return {
        "issue_id": issue.get("issue_id"),

        "issue_type": issue.get("issue_type"),

        "issue_name": issue.get(
            "cluster_name",
            issue.get("issue_type")
        ),

        "severity": issue.get("severity"),

        "deterministic_facts": {
            "affected_rows": _affected_rows(issue),
            "evidence": issue.get("evidence"),
            "rule_triggered": issue.get("rule_triggered"),
            "detector_confidence": issue.get("confidence"),
            "possible_root_cause_category": issue.get(
                "possible_root_cause_category"
            ),
        },

        "ai_interpretation": {
            "issue_summary": investigation.get(
                "issue_summary"
            ),
            "likely_root_cause": investigation.get(
                "likely_root_cause"
            ),
            "interpretation": investigation.get(
                "interpretation"
            ),
            "recommendation": investigation.get(
                "recommendation"
            ),
            "confidence": investigation.get(
                "confidence"
            ),
            "assumptions": investigation.get(
                "assumptions",
                []
            ),
        },

        "source": investigation.get(
            "source",
            "unknown"
        ),
    }