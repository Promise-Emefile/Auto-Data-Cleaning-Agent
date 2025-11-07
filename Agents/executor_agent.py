import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
import aisuite as ai
import re

load_dotenv()
client = ai.Client()

def execute_generated_code(redefined_code: str, df: pd.DataFrame):
    """
    Executes AI-generated cleaning code safely within a controlled environment.
    Returns the cleaned DataFrame or None if execution fails.
    """
    match = re.search(r"<execute>([\s\S]*?)</execute>", redefined_code)
    if match:
        code = match.group(1).strip()
    else: 
        code = redefined_code.strip()
        try:
            exec_globals = {"df": df.copy()}
            exec(code, exec_globals)  # run code in isolated context
            print("Code executed successfully.")
            cleaned_df = exec_globals.get("df", None)
            return cleaned_df
        except Exception as e:
            print(f"Execution error: {e}")
            return None
    else:
        print("No valid <execute> block found.")
        return None


