import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
import aisuite as ai

load_dotenv()
client = ai.Client()

def build_code_gen(plan, df=None):
    """Builds a code generation propmt based on planner output and optional dataframe info."""
    column_info = ""
    if df is not None:
        column_info = f"\n\nThe dataset has the following columns:\n{df.dtypes.to_dict()}"
    prompt = f"""
    You are a CODE GENERATOR AI AGENT.

    You are given:
    1. A pandas DataFrame named df
    2. A list of steps (the cleaning plan) created by another AI planner

    PLANNER STEPS:
    {plan}

    {column_info}

    Your task:
    - Translate each planner step into *fully functional, safe, and efficient Python code* using pandas.
    - Ensure that all operations are compatible with the actual dtypes.
    - Apply these cleaning principles:
      * Handle missing values:
          - Numeric → fill with mean
          - Categorical → fill with mode
      * Drop duplicates
      * Handle outliers (detect, but do NOT delete)
      * Avoid inplace=True misuse
      * Avoid chained assignments
      * Avoid using .str on non-string columns
      * Avoid deprecated pandas functions

    Output only valid Python code wrapped in <execute>...</execute> tags.
    The code must assign the final cleaned DataFrame back to df.
    """
    return prompt

def code_gen_agent(plan, df=None):
    prompt = build_code_gen(plan, df)

    response = client.chat.completions.create(
        model="openai:gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a Code Generator AI Agent."},
            {"role": "user", "content": prompt}
        ]
    )
    code = response.choices[0].message.content.strip()
    return code



