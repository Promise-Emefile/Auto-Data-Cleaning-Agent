import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
import aisuite as ai

load_dotenv()
client = ai.Client()

def prompt_for_critic(code_output, plan=None, summary=None, columns=None):
    plan_text = f"\nCleaning Plan:\n{plan}" if plan else ""
    summary_text = f"\nDataset Summary:\n{summary}" if summary else ""
    columns_text = f"\nColumns: {columns}" if columns else ""

    prompt = f"""
    You are a Senior Data Cleaning Agent.
    {summary_text}
    {plan_text}
    {columns_text}

    Your Task:
    You are to review this code, critique it, and improve it while maintaining the same logic.
    - Check for errors or inefficiencies.
    - PRESERVE categorical columns exactly as text (no binary/numeric conversion).
    - Clean categorical inconsistencies by lowercasing and trimming strings safely:
        * Check dtype before using .str methods.
        * Avoid inplace=True and chained assignment warnings.
    - Ensure the final DataFrame has the same number of rows as the original.
    - Retain the original dataset shape.
    - Follow the provided cleaning plan.
    - Always assign the final cleaned DataFrame back to df.

    Code to review:
    {code_output}

    Return only the improved Python code wrapped inside:
    <execute>
        # improved Python code
    </execute>
    """
    return prompt

def critic_code(code_output, plan=None, summary=None, columns=None):
    prompt = prompt_for_critic(code_output, plan, summary, columns)
    response = client.chat.completions.create(
        model="openai:gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a meticulous AI code reviewer."},
            {"role": "user", "content": prompt}
        ]
    )
    critic_gen_code = response.choices[0].message.content.strip()
    return critic_gen_code



