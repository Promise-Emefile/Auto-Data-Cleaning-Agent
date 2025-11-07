import streamlit as st
import pandas as pd
import sys
import os
import aisuite as ai

client = ai.Client()

# Ensure current folder and "agents" subfolder are on the path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

# Import your actual agent functions
from Agents.planning_agent import planner_agent, build_planner_prompt, build_dataset_summary
from Agents.Code_Generating_Agent import build_code_gen, code_gen_agent
from Agents.Critic_agent import prompt_for_critic, critic_code
from Agents.executor_agent import execute_generated_code
from Agents.Validation_agent import llm_validation_report, programmatic_validation

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
            summary = build_dataset_summary(df)
            plan = planner_agent(summary)
        st.success("Cleaning plan generated successfully!")
        st.code(plan, language="markdown")

        # Step 3: Code generation
        with st.spinner("Generating data cleaning code..."):
            code_output = code_gen_agent(plan, df)
        st.success("Code generated successfully!")
        st.code(code_output, language="python")

        # Step 4: Critic refinement
        with st.spinner("Critic agent reviewing and refining code..."):
            refined_code = critic_code(code_output)
        st.success("Critic agent produced refined code!")
        st.code(refined_code, language="python")

        # Step 5: Execute refined code
        with st.spinner("Executing refined code..."):
            cleaned_df = execute_generated_code(refined_code, df)

        if cleaned_df is not None:
            st.success("Data cleaned successfully!")
            st.dataframe(cleaned_df.head())

            # Step 6: Validation
            with st.spinner("Validating cleaned data (programmatic + LLM)..."):
                validation_prog = programmatic_validation(cleaned_df)
                validation_llm = llm_validation_report(client, cleaned_df, validation_prog)

            st.write("### Programmatic Validation Report", validation_prog)
            st.json(validation_llm)

            # Step 7: Feedback loop — if validation failed, trigger focused re-cleaning
            if validation_prog["validation_result"].startswith("Fail"):
                st.warning("Validation failed — initiating targeted re-cleaning...")

                from Agents.feedback_agent import feedback_to_plan

                # Generate a focused repair plan
                with st.spinner("Generating repair plan from validation feedback..."):
                    feedback_plan = feedback_to_plan(validation_prog, cleaned_df)
                st.success("Repair plan generated successfully!")
                st.json(feedback_plan)

                # Generate refined code using same code generator
                with st.spinner("Generating targeted re-cleaning code..."):
                    re_clean_code = code_gen_agent(feedback_plan, cleaned_df)
                st.code(re_clean_code, language="python")

                # Execute re-cleaning
                with st.spinner("⚙ Running targeted re-cleaning..."):
                    cleaned_df_v2 = execute_generated_code(re_clean_code, cleaned_df)

                if cleaned_df_v2 is not None:
                    st.success("Re-cleaning successful!")
                    st.dataframe(cleaned_df_v2.head())

                    # Optional re-validation
                    final_validation = programmatic_validation(cleaned_df_v2)
                    st.write("### Final Validation Report", final_validation)

                    # Add download option after re-cleaning
                    csv_data_v2 = cleaned_df_v2.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇ Download Re-Cleaned Dataset",
                        data=csv_data_v2,
                        file_name="recleaned_dataset.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Re-cleaning execution failed. Check generated code or logs.")

