import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
from openai import OpenAI

# Import your custom backend engines
from src.analytics import load_and_optimize_data, get_top_sponsors_by_state_and_role, get_employer_analytics
from src.rag import query_compliance_engine

# Load environment variables (API Keys)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Joblore | Career Intelligence", page_icon="🧭", layout="wide")
st.title("🧭 Joblore")
st.markdown("Eliminating information asymmetry for international students navigating the U.S. labor market.")

# --- 2. CACHING THE DATA ENGINE ---
@st.cache_data
def init_data():
    """Loads the 500k+ row dataset into RAM only once to keep the app blazing fast."""
    return load_and_optimize_data()

with st.spinner("⏳ Booting up the Joblore Data Engine..."):
    master_df = init_data()

# --- 3. UI LAYOUT: TABS ---
tab1, tab2, tab3 = st.tabs(["🌎 Discovery Hub", "🏢 Employer Profiles", "⚖️ Compliance Assistant"])

# === TAB 1: MACRO DISCOVERY ===
with tab1:
    st.header("Search Top Sponsors by Region & Role")
    
    col1, col2 = st.columns(2)
    with col1:
        target_state = st.text_input("State Code (e.g., IL, CA, NY)", value="IL")
    with col2:
        target_role = st.text_input("Job Role (e.g., DATA, SOFTWARE)", value="DATA")

    if st.button("Search Market"):
        results = get_top_sponsors_by_state_and_role(master_df, target_state, target_role)
        
        if results.empty:
            st.warning("No records found for that combination.")
        else:
            # Display data table
            st.dataframe(results, use_container_width=True)
            
            # Display beautiful interactive bar chart
            fig = px.bar(
                results, 
                x='EMPLOYER_NAME', 
                y='Sponsorship_Count', 
                title=f"Top Sponsors for {target_role.upper()} in {target_state.upper()}",
                color='Median_Salary',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)


# === TAB 2: MICRO EMPLOYER PROFILES ===
with tab2:
    st.header("Deep Dive: Company Sponsorship Profile")
    company_search = st.text_input("Enter Company Name (e.g., GOOGLE LLC)", value="GOOGLE LLC")
    
    if st.button("Analyze Employer"):
        profile = get_employer_analytics(master_df, company_search)
        
        if "error" in profile:
            st.error(profile["error"])
        else:
            # Big metric numbers
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Total Visas Sponsored", profile["total_sponsorships"])
            metric_col2.metric("Median Salary", f"${profile['median_salary']:,.2f}")
            
            # Split lists side-by-side
            list_col1, list_col2 = st.columns(2)
            with list_col1:
                st.subheader("Top Sponsored Roles")
                st.write(profile["top_roles"])
            with list_col2:
                st.subheader("Top Geographic Hubs")
                st.write(profile["top_states"])


# === TAB 3: AI COMPLIANCE ASSISTANT (RAG) ===
with tab3:
    st.header("⚖️ Legal & Policy Assistant")
    st.markdown("Ask questions regarding CPT, OPT, and STEM Extensions.")
    
    school_selection = st.selectbox("Select Your University", ["uiuc", "purdue", "federal"])
    user_question = st.text_area("What do you need help with?")
    
    if st.button("Ask Joblore AI"):
        if not user_question:
            st.warning("Please enter a question.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.error("Missing OpenAI API Key! Please create a .env file and add your key.")
        else:
            with st.spinner("Scanning compliance documents and generating response..."):
                # Step A: Retrieve the math-matched text chunks from ChromaDB
                matched_chunks = query_compliance_engine(school_selection, user_question)
                
                # Step B: Combine chunks into a single "Context" string
                context_string = "\n\n".join([chunk.page_content for chunk in matched_chunks])
                
                # Step C: Prompt Engineering - Force the AI to read ONLY our documents
                ai_prompt = f"""
                You are a strict immigration compliance assistant for international students.
                Use ONLY the following context to answer the user's question. 
                If the answer is not contained in the context, say "I cannot find the official rule for this in my database."
                
                CONTEXT:
                {context_string}
                
                USER QUESTION: 
                {user_question}
                """
                
                # Step D: Send it to OpenAI
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": ai_prompt}],
                    temperature=0.2
                )
                
                # Display the AI's final answer
                st.success("Analysis Complete!")
                st.write(response.choices[0].message.content)
                
                # Show the citations so the student knows it's true
                with st.expander("🔍 View Source Documents Used"):
                    for i, chunk in enumerate(matched_chunks, 1):
                        st.caption(f"Source {i}: {chunk.metadata.get('file_name')} ({chunk.metadata.get('source_scope').upper()})")