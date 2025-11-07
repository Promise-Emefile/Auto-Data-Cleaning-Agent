import streamlit as st
import pandas as pd
import sys
import os
import json
import datetime
import aisuite as ai

# Initialize LLM client
client = ai.Client()

# Ensure correct module paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

# Import agent modules
from Agents.planning_agent import planner_agent, build_dataset_summary
from Agents.Code_Generating_Agent import code_gen_agent
from Agents.Critic_agent import critic_code
from Agents.executor_agent import execute_generated_code
from Agents.Validation_agent import programmatic_validation, llm_validation_report

# Setup Streamlit app
st.set_page_config(page_title="Auto Data Cleaning Agent", layout="wide")
st.title("Auto Data Cleaning Agent")
st.caption("A multi-agent LLM-powered pipeline for autonomous, iterative data cleaning and validation.")

# Create log directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

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
            st.success("Initial cleaning complete!")
            st.dataframe(cleaned_df.head())

            # Step 6: Iterative Validation and Feedback Loop
            max_attempts = 3
            attempt = 1
            current_df = cleaned_df
            from Agents.feedback_agent import feedback_to_plan

            while attempt <= max_attempts:
                st.write(f"### Validation Attempt {attempt}")
                with st.spinner("Validating cleaned data (programmatic + LLM)..."):
                    validation_prog = programmatic_validation(current_df)
                    validation_llm = llm_validation_report(client, current_df, validation_prog)

                st.write("#### Programmatic Validation Report:")
                st.json(validation_prog)
                st.write("#### LLM Validation Report:")
                st.json(validation_llm)

                # Log the attempt
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = os.path.join(LOG_DIR, f"attempt_{attempt}_{timestamp}.json")
                with open(log_path, "w") as f:
                    json.dump({
                        "attempt": attempt,
                        "validation_prog": validation_prog,
                        "validation_llm": validation_llm
                    }, f, indent=2)

                # Check validation result
                if validation_prog["validation_result"].startswith("Pass"):
                    st.success(f"Data passed validation on attempt {attempt}!")
                    final_df = current_df
                    break
                else:
                    st.warning(f"Validation failed — triggering re-cleaning (attempt {attempt})...")
                    with st.spinner("Generating focused repair plan..."):
                        feedback_plan = feedback_to_plan(validation_prog, current_df)
                    st.json(feedback_plan)

                    with st.spinner("Generating targeted re-cleaning code..."):
                        re_clean_code = code_gen_agent(feedback_plan, current_df)
                    st.code(re_clean_code, language="python")

                    with st.spinner("Executing targeted re-cleaning..."):
                        new_df = execute_generated_code(re_clean_code, current_df)

                    if new_df is None:
                        st.error("Re-cleaning execution failed. Stopping loop.")
                        final_df = current_df
                        break
                    else:
                        st.success("Re-cleaning round completed.")
                        current_df = new_df
                        attempt += 1

            else:
                st.warning("Maximum re-cleaning attempts reached. Some issues may remain.")
                final_df = current_df

            # Step 7: Final output
            st.write("### Final Cleaned Dataset Preview")
            st.dataframe(final_df.head())

            csv_data = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Final Cleaned Dataset",
                data=csv_data,
                file_name="final_cleaned_dataset.csv",
                mime="text/csv"
            )

            st.info("Logs saved in /logs folder for debugging and reproducibility.")
        else:
            st.error("Cleaning failed. Check console or logs for details.")



