import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
import aisuite as ai

load_dotenv()
client = ai.Client()

import re
def execute_generated_code(redefined_code: str):
    # Extract code from <execute_python> tags
    match = re.search(r"<execute>([\s\S]*?)</execute>", redefined_code)
    if match:
        code = match.group(1).strip()
        try:
            exec_globals = {}
            exec(code, globals())  # Run the code in global scope
            print("Code executed successfully.")
            return exec_globals.get("df", None)
        except Exception as e:
            print(f" Execution error: {e}")
            return None
    else:
        print(" No valid <execute> block found.")
        return None

  execute_generated_code(redefined_code)
