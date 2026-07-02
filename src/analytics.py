import pandas as pd
import os

# Define the absolute path to your data file
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Combined_LCA_Disclosure_Data_FY2020_to_FY2024.csv")

def load_and_optimize_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find the dataset at {DATA_PATH}. Ensure it is named Combined_LCA_Disclosure_Data_FY2020_to_FY2024.csv")
    
    print("⏳ Analyzing CSV schema and column headers...")
    
    # Read just the first row to inspect what headers are used in this version
    preview_df = pd.read_csv(DATA_PATH, nrows=1)
    available_columns = preview_df.columns.tolist()
    
    # Dynamic Mapping Strategy: Find the right columns even if names vary slightly
    column_mapping = {}
    
    # 1. Match Case Status
    status_col = [c for c in available_columns if 'STATUS' in c.upper()]
    column_mapping['CASE_STATUS'] = status_col[0] if status_col else None
    
    # 2. Match Employer Name
    emp_col = [c for c in available_columns if 'EMPLOYER' in c.upper() and 'NAME' in c.upper()]
    if not emp_col: # Fallback to any column with EMPLOYER
        emp_col = [c for c in available_columns if 'EMPLOYER' in c.upper()]
    column_mapping['EMPLOYER_NAME'] = emp_col[0] if emp_col else None
    
    # 3. Match Job Title
    job_col = [c for c in available_columns if 'JOB' in c.upper() and 'TITLE' in c.upper()]
    column_mapping['JOB_TITLE'] = job_col[0] if job_col else None
    
    # 4. Match Wage
    wage_col = [c for c in available_columns if 'WAGE' in c.upper()]
    column_mapping['PREVAILING_WAGE'] = wage_col[0] if wage_col else None
    
    # 5. Match Worksite State
    state_col = [c for c in available_columns if 'STATE' in c.upper() and 'WORK' in c.upper()]
    if not state_col: # Fallback to any state column
        state_col = [c for c in available_columns if 'STATE' in c.upper()]
    column_mapping['WORKSITE_STATE'] = state_col[0] if state_col else None

    # Check if we missed anything critical
    missing = [k for k, v in column_mapping.items() if v is None]
    if missing:
        print(f"⚠️ Warning: Could not automatically map columns for: {missing}")
        print(f"Available headers in your file are: {available_columns}")
        # Let's apply standard default assumptions as a fallback
        column_mapping['CASE_STATUS'] = 'CASE_STATUS'
        column_mapping['EMPLOYER_NAME'] = 'EMPLOYER_NAME' if 'EMPLOYER_NAME' in available_columns else 'EMPLOYER_BUSINESS_NAME'
        column_mapping['JOB_TITLE'] = 'JOB_TITLE'
        column_mapping['PREVAILING_WAGE'] = 'PREVAILING_WAGE'
        column_mapping['WORKSITE_STATE'] = 'WORKSITE_STATE'

    print(f"⚙️ Mapping identified columns: {column_mapping}")
    
    # Now read the full file using our safe matched columns
    read_cols = [v for v in column_mapping.values() if v in available_columns]
    
    print("⏳ Parsing rows and filtering for certified applications...")
    df = pd.read_csv(DATA_PATH, usecols=read_cols)
    
    # Standardize column names back to our internal Joblore system format
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    df = df.rename(columns=reverse_mapping)
    
    # Filter for approved cases only
    df = df[df['CASE_STATUS'].astype(str).str.upper().str.strip() == 'CERTIFIED']
    
    # Clean text metrics
    df['EMPLOYER_NAME'] = df['EMPLOYER_NAME'].astype(str).str.upper().str.strip()
    df['JOB_TITLE'] = df['JOB_TITLE'].astype(str).str.upper().str.strip()
    df['WORKSITE_STATE'] = df['WORKSITE_STATE'].astype(str).str.upper().str.strip()
    
    # Clean numeric metrics
    df['PREVAILING_WAGE'] = pd.to_numeric(df['PREVAILING_WAGE'], errors='coerce')
    df = df.dropna(subset=['PREVAILING_WAGE'])
    
    print(f"✅ Loaded {len(df):,} certified records into Joblore analytics engine core!")
    return df

# FEATURE 1: Discovery Search (By State & Role Keyword)
def get_top_sponsors_by_state_and_role(df, state_code, search_role):
    # Filter by target state (handle both full state names or two-letter codes if raw text is messy)
    filtered_df = df[df['WORKSITE_STATE'].str.contains(state_code.upper(), na=False)]
    
    # Filter by job title matching keyword
    filtered_df = filtered_df[filtered_df['JOB_TITLE'].str.contains(search_role.upper(), na=False)]
    
    if filtered_df.empty:
        print(f"ℹ️ No matching records found for Role: {search_role} in State: {state_code}")
        return pd.DataFrame()
        
    # Aggregate counts and median salary
    summary = filtered_df.groupby('EMPLOYER_NAME').agg(
        Sponsorship_Count=('CASE_STATUS', 'count'),
        Median_Salary=('PREVAILING_WAGE', 'median')
    ).reset_index()
    
    top_10 = summary.sort_values(by='Sponsorship_Count', ascending=False).head(10)
    top_10['Median_Salary'] = top_10['Median_Salary'].round(2)
    
    return top_10

# FEATURE 2: Specific Company Profile Lookup
def get_employer_analytics(df, employer_name):
    emp_df = df[df['EMPLOYER_NAME'] == employer_name.upper().strip()]
    
    if emp_df.empty:
        return {"error": f"No certified records found for employer: {employer_name}"}
        
    total_sponsorships = len(emp_df)
    median_salary = emp_df['PREVAILING_WAGE'].median()
    
    # Extract top 5 jobs and top 3 geographic hubs
    top_roles = emp_df['JOB_TITLE'].value_counts().head(5).to_dict()
    top_states = emp_df['WORKSITE_STATE'].value_counts().head(3).to_dict()
    
    return {
        "employer": employer_name.upper(),
        "total_sponsorships": total_sponsorships,
        "median_salary": round(median_salary, 2),
        "top_roles": top_roles,
        "top_states": top_states
    }


if __name__ == "__main__":
    try:
        master_df = load_and_optimize_data()
        
        print("\n🔍 Testing Feature 1 [Role: DATA, State: IL]...")
        test_discover = get_top_sponsors_by_state_and_role(master_df, "IL", "DATA")
        print(test_discover.to_string(index=False))
        
        print("\n🔍 Testing Feature 2 [Company Profile: GOOGLE LLC]...")
        test_profile = get_employer_analytics(master_df, "GOOGLE LLC")
        print(test_profile)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")