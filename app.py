import streamlit as st
import pandas as pd
from Agents.planning_agent import planning_agent
from Agents.Code_Generating_agent import Code_Generating_Agent
from Agents.critic_agent import critic_agent
from Agents.executor_agent import execute_code
from Agents.Validation_agent import Validation_agent

# Set page layout and metadata
st.set_page_config(page_title="Auto Data Cleaning Agent", layout="wide")

# Page header
st.title("Auto Data Cleaning Agent")
st.caption("LLM-powered multi-agent pipeline for autonomous data cleaning and validation")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file to clean", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Dataset", df.head())

    if st.button("Run Auto Cleaning Pipeline"):
        with st.spinner("Planning cleaning steps..."):
            plan = planning_agent(df)
        st.success("Plan generated successfully!")
        st.code(plan, language="markdown")

        with st.spinner("Generating cleaning code..."):
            gen_code = Code_Generating_agent(plan)
        st.success("Code generated!")
        st.code(gen_code, language="python")

        with st.spinner("Refining with critic agent..."):
            critic_code = critic_agent(gen_code)
        st.success("Critic produced refined code!")
        st.code(critic_code, language="python")

        with st.spinner("Executing refined code..."):
            execute_generated_code(critic_code)

        # After execution, check if df exists in globals
        if 'df' in globals():
            cleaned_df = globals()['df']
            st.success("Execution successful!")
            st.write("### Cleaned Dataset Preview", cleaned_df.head())

            with st.spinner("Validating cleaned data..."):
                validation = validation_agent(cleaned_df)
            st.write("### Validation Report", validation)
        else:
            st.error("No cleaned dataframe detected after execution. Check critic output for issues.")

