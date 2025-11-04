import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
import aisuite as ai
import display_functions

load_dotenv()
client = ai.Client()


def programmatic_validation(df):
    issues = []
    warnings = []  # NEW: track warnings separately (for outliers)

    # Check for missing values
    missing_summary = df.isnull().sum()
    if missing_summary.sum() > 0:
        issues.append(f"Missing values found in columns: {missing_summary[missing_summary > 0].to_dict()}")

    # Check boolean columns ('fbs', 'exang')
    for col in ["fbs", "exang"]:
        if col in df.columns:
            if df[col].dtype != "bool" and not (
                str(df[col].dtype).startswith("category") and
                set(df[col].cat.categories) <= {True, False}
            ):
                issues.append(f"Column '{col}' not boolean (current dtype: {df[col].dtype})")

    # Check outliers (simple heuristic)
    for col in ["trestbps", "chol"]:
        if col in df.columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
            if outliers > 0:
                # Instead of "issue", record as a "warning"
                warnings.append(f"Potential outliers detected in '{col}' ({outliers} rows)")

    # Final result logic
    if issues:
        validation_result = "Fail "
    elif warnings:
        validation_result = "Warning "
    else:
        validation_result = "Pass "

    # Build result JSON
    return {
        "validation_result": validation_result,
        "issues_found": issues,
        "warnings_found": warnings,
        "summary": (
            "No issues detected."
            if not issues and not warnings
            else (
                "Outliers detected — review but likely safe to proceed."
                if warnings and not issues
                else "Critical issues found — requires re-cleaning."
            )
        )
    }


def llm_validation_report(client, df, programmatic_report):
    prompt = f"""
    You are a data validation agent.
    Here is the dataframe summary:
    shape: {df.shape}
    dtypes: {df.dtypes.to_dict()}
    validation_check: {programmatic_report}

    Based on this, produce a concise JSON report with:
    - validation_result ("Pass" or "Fail")
    - issues_found (list)
    - recommendation
    """

    response = client.chat.completions.create(
        model="openai:gpt-4o-mini",
        temperature = 0,
        messages=[
            {"role": "system", "content": "You are a careful data validation agent."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content

programmatic_report = programmatic_validation(df)
final_validation = llm_validation_report(client, df, programmatic_report)

print(final_validation)
