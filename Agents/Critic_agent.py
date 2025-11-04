import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
import aisuite as ai

load_dotenv()
client = ai.Client()


def prompt_for_critic(generated_code):
    prompt = f"""
    You are a Senior Data Cleaning Agent.
    Here is the DataFrame info summary{summary}
Your Task:
You are to review this code{generated_code}, critise the code while still following the plan{plan}.
- Check for errors.
- PRESERVE categorical columns exactly as text (no binary/numeric conversion).
- Cleans categorical inconsistencies by lowercasing and trimming strings safely:
   - Checks dtype before using .str methods.
   - Avoids inplace=True and chained assignment warnings.
   - Logs every major step with print statements.
- Ensure the final DataFrame has the same number of rows as the original.
- Retain the original datashape.
- follow the plan{plan}
- You are given a pandas DataFrame called df
- The code must assign the final cleaned DataFrame back to df.
columns:{columns}

Return the output code in the format <execute> Python code</execute>

"""
    return prompt

def critic_code(generated_code):
    prompt = prompt_for_critic(generated_code)
    response = client.chat.completions.create(
    model = "openai:gpt-4o-mini",
    temperature=0,
    messages = [
        {"role":"system", "content":"You are a helpful assistant"},
        {"role":"user", "content":prompt}
    ])
    critic_gen_code = response.choices[0].message.content.strip()
    return critic_gen_code


