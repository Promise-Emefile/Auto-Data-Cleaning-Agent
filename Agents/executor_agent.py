import pandas as pd
import numpy as np
import aisuite as ai
import re
import traceback
from dotenv import load_dotenv

load_dotenv()
client = ai.Client()

def execute_generated_code(refined_code: str, df: pd.DataFrame):
    """
    Executes refined cleaning code safely on a given DataFrame and returns the cleaned version.
    """
    # Extract code block between <execute> tags
    match = re.search(r"<execute>([\s\S]*?)</execute>", refined_code)
    if not match:
        print("No valid <execute> block found.")
        return None

    code = match.group(1).strip()
    print("Extracted code for execution:\n", code[:300], "..." if len(code) > 300 else "")

    # Create a local environment with a copy of df
    local_env = {"df": df.copy()}

    try:
        exec(code, {}, local_env)  # Execute safely within a local namespace
        if "df" not in local_env:
            raise ValueError("The executed code did not return a variable named 'df'.")
        print("Code executed successfully.")
        return local_env["df"]
    except Exception as e:
        print("Execution error:", e)
        traceback.print_exc()
        return None


