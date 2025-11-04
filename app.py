import streamlit as st
import pandas as pd
import sys
import os

# Ensure the "agents" folder is discoverable
sys.path.append(os.path.dirname(_file_))  # ensure current folder on path
sys.path.append(os.path.join(os.path.dirname(_file_), "agents"))

# Import your actual agent functions
from agents.planner_agent import planner_agent, build_planner_prompt
from agents.code_gen_agent import build_code_gen, code_gen_agent
from agents.critic_agent import prompt_for_critic, critic_code
from agents.executor_agent import execute_generated_code
from agents.validation_agent import llm_validation_report, programmatic_validation

# Streamlit page setup
st.set_page_config(
    page_title="Auto Data Cleaning Agent",
    layout="wide"
)

st.title("Auto Data Cleaning Agent")
st.caption("A multi-agent LLM-powered pipeline for autonomous data cleaning and validation.")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Dataset Preview", df.head())

    if st.button("Run Auto Cleaning Pipeline"):
        # Step 2: Planning
        with st.spinner("Generating cleaning plan..."):
            plan = planner_agent(df)
        st.success("Cleaning plan generated successfully!")
        st.code(plan, language="markdown")

        # Step 3: Code generation
        with st.spinner("Generating data cleaning code..."):
            code_output = code_gen_agent(plan)
        st.success("Code generated successfully!")
        st.code(code_output, language="python")

        # Step 4: Critic refinement
        with st.spinner("Critic agent reviewing and refining code..."):
            refined_code = critic_code(code_output)
        st.success("Critic agent produced refined code!")
        st.code(refined_code, language="python")

        # Step 5: Execute refined code
        with st.spinner("Executing refined code..."):
            cleaned_df = execute_generated_code(refined_code)

        if cleaned_df is not None:
            st.success("Code executed successfully!")
            st.write("### Cleaned Dataset Preview", cleaned_df.head())

            # Step 6: Validation
            with st.spinner("Validating cleaned data (programmatic + LLM)..."):
                validation_prog = programmatic_validation(cleaned_df)
                validation_llm = llm_validation_report(cleaned_df)

            st.write("### Programmatic Validation Report", validation_prog)

