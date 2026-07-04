# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the Joblore repository.

## 🚀 Development Commands

### Setup
1. Create a virtual environment (if not already present):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   - Copy `.env.example` to `.env` (if it exists) or create a `.env` file
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`
   - **Note**: The `.env` file is already present and gitignored. Ensure it contains a valid OpenAI API key for the AI Compliance Assistant tab to function.

### Running the Application
```bash
streamlit run src/app.py
```
This launches the Streamlit web interface at http://localhost:8501.

### Data & Dependencies
- The primary dataset (`data/Combined_LCA_Disclosure_Data_FY2020_to_FY2024.csv`) is large (~2.8GB) and is intentionally excluded from version control via `.gitignore`.
- The vector database (`chroma_db/`) is also excluded and will be generated automatically when the RAG tab is first used.
- Compliance PDFs in `/docs/` are excluded from version control but are required for the RAG functionality.

## 🏗️ Project Architecture

### High-Level Components
1. **Data Engine** (`src/analytics.py`):
   - Loads and optimizes the H-1B LCA disclosure dataset.
   - Provides functions for:
     - `load_and_optimize_data()`: Efficiently loads CSV with dynamic column mapping.
     - `get_top_sponsors_by_state_and_role()`: Returns top sponsors for a given state/job role.
     - `get_employer_analytics()`: Returns sponsorship statistics for a specific employer.
   - **Job title normalization**: Raw job titles are cleaned (e.g., converting ".NET" to "DOT NET", "C#" to "C SHARP") to reduce variants, and the dropdown is limited to the top 20 most frequent roles for better performance.

2. **Retrieval-Augmented Generation (RAG) Engine** (`src/rag.py`):
   - Uses `sentence-transformers` to create embeddings of compliance PDFs.
   - Stores embeddings in a persistent ChromaDB vector store (`chroma_db/`).
   - Retrieves relevant document chunks based on user query and school scope (federal/university-specific).
   - Uses OpenAI's GPT-4o-mini to generate answers strictly based on retrieved context.

3. **Streamlit Web Application** (`src/app.py`):
   - Three-tab interface:
     - **Discovery Hub**: Search top sponsors by state and role (uses analytics module).
     - **Employer Profiles**: Deep dive into a company's sponsorship history (uses analytics module).
     - **AI Compliance Assistant**: Ask questions about CPT/OPT/STEM extensions (uses RAG module).
   - Features:
     - Data caching with `@st.cache_data` for fast reloads.
     - Interactive visualizations with Plotly.
     - Secure handling of API keys via environment variables.

### Key Directories
- `src/` - Main application source code
  - `app.py` - Streamlit entry point
  - `analytics.py` - Data processing engine
  - `rag.py` - RAG pipeline for compliance questions
- `data/` - Contains the large H-1B dataset (gitignored)
- `docs/` - Contains compliance PDFs (federal and university-specific, gitignored)
- `chroma_db/` - Persistent vector store for RAG (generated on first use, gitignored)
- `.venv` - Python virtual environment (gitignored)

### Important Notes
- The application is designed to run entirely locally with no external dependencies beyond the Python packages in `requirements.txt`.
- All sensitive data (API keys) are managed via environment variables.
- Data loading is optimized with Streamlit's caching to ensure fast user interactions after initial load.
- The RAG system is designed to be strictly grounded in the provided compliance documents, reducing hallucination risk.

## 🔧 Development Guidelines
When modifying this codebase:
1. **Data Engine Changes**: If modifying `analytics.py`, ensure the dynamic column mapping remains robust to variations in the H-1B CSV format. Note that job titles are now normalized (e.g., converting ".NET" to "DOT NET", "C#" to "C SHARP") to reduce variants, company names are filtered to remove non‑company entries (e.g., phone numbers), and both role and company dropdowns are limited to the top 20 most frequent roles and top 50 most frequent companies for better performance.
2. **RAG Changes**: If updating `rag.py`, test that the vector store regenerates correctly and that source filtering (federal vs. university) works as expected.
3. **UI Changes**: Keep the Streamlit interface intuitive and responsive. Use caching appropriately to maintain performance.
4. **Environment**: Never commit `.env` or any files containing secrets. The `.gitignore` already excludes these.
5. **Dependencies**: Keep `requirements.txt` updated if adding new packages.