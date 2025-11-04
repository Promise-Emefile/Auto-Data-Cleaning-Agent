import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import aisuite as ai
import display_functions

load_dotenv()
client = ai.Client()

columns = df.columns

def build_code_gen(plan):
    prompt = f"""
    You are a CODE GENERATOR AI AGENT.

You are given:
1. A pandas DataFrame named df
2. A list of steps (the cleaning plan) created by another AI planner

PLANNER STEPS:
{plan}

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
The code must assign the final cleaned DataFrame back to df.
"""
    return prompt


def code_gen_agent(plan):
    prompt = build_code_gen(plan)

    response = client.chat.completions.create(
        model = "openai:gpt-4o-mini",
        temperature=0,
        messages = [
            {"role": "system", "content": "You are a Code Generator AI Agent."},
            {"role": "user", "content": prompt}
        ]
    )
    code = response.choices[0].message.content.strip()
    return code
  generated_code = code_gen_agent(plan)
print(generated_code)
