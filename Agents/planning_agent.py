import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
import aisuite as ai

load_dotenv()
client = ai.Client()

summary = {
    "shape": df.shape,
    "missing_values": df.isna().sum().to_dict(),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "duplicates": df.duplicated().sum()
}

def build_planner_prompt(summary):
    prompt = f"""
    You are a data quality planning agent.
    Analyze the dataset summary below and plan the cleaning steps required.
    Be specific about which columns need fixing and what actions to take.

    Dataset Summary:
    {summary}

    Respond in this format:
    {{
    "actions":["...","..."],
    "priority_order":["..."]
    }}
    """
    return prompt


import json
import requests

def planner_agent(summary):
    prompt = build_planner_prompt(summary)

    response = client.chat.completions.create(
        model="openai:gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an expert data planner agent."},
            {"role": "user", "content": prompt}
        ]
    )

    reply = response.choices[0].message.content.strip()

    # Parse output
    try:
        plan = json.loads(reply)
    except json.JSONDecodeError:
        plan = {"actions": [reply], "priority_order": []}

    return plan

plan= planner_agent(summary)
plan
