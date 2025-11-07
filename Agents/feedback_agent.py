import json
from dotenv import load_dotenv
import aisuite as ai

load_dotenv()
client = ai.Client()

def feedback_to_plan(validation_output, df):
    """
    Takes validation report and half-cleaned DataFrame,
    and generates a focused re-cleaning plan targeting only failed areas.
    """
    prompt = f"""
    You are a Data Repair Planner Agent.

    A previous data cleaning process partially succeeded, but validation failed.

    Validation Report:
    {json.dumps(validation_output, indent=2)}

    Dataset Info:
    shape: {df.shape}
    columns: {df.columns.tolist()}
    dtypes: {df.dtypes.astype(str).to_dict()}

    Task:
    - Create a precise repair plan to fix ONLY the failed columns or problems listed above.
    - DO NOT repeat operations on already cleaned columns.
    - If missing values are detected → handle them properly (mean/mode based on type).
    - If dtype mismatch is found → correct it safely.
    - If inconsistencies in categories exist → clean them without converting to numeric.
    - Return your response as JSON with fields:
        {{
            "actions": [...],
            "priority_order": [...]
        }}
    """

    response = client.chat.completions.create(
        model="openai:gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a meticulous data repair planner."},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        plan = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        plan = {"actions": [response.choices[0].message.content], "priority_order": []}

    return plan

