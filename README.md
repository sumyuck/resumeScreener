# Resume Screener

An AI-powered resume screening and candidate ranking tool that automates initial recruitment workflows. Parses resumes, extracts structured fields, scores candidates against job descriptions using a hybrid RAG pipeline, and provides evidence-backed justifications.

Built for the Sprinto AI Implementation Intern Assignment.

## Features

### Core
- **Multi-format parsing**: PDF and DOCX support with PyPDF2 + pdfplumber fallback
- **Dynamic extraction**: Configurable field extraction (name, skills, experience, etc.) via UI
- **AI scoring with justification**: 1-10 fit score with LLM-generated explanations
- **Duplicate detection**: Exact (SHA-256 hash) + fuzzy (embedding similarity) detection
- **Batch re-parsing**: Re-scan all resumes when extraction config changes

### Enhanced
- **Hybrid RAG pipeline**: pgvector semantic search + keyword matching with reciprocal rank fusion
- **Evidence-backed scoring**: Each score cites retrieved chunks from the resume
- **Requirement weighting**: Must-have (3x), Good-to-have (2x), Bonus (1x)
- **Multi-role matching**: Score one candidate across multiple JDs
- **Recruiter feedback**: Shortlist / Reject / Maybe workflow per candidate per role
- **Interview question generation**: AI-generated questions targeting scoring gaps
- **Resume quality checks**: Completeness and quality assessment
- **Scan history**: Full audit trail of all scan runs
- **Delete support**: Delete resumes and roles with cascade cleanup and confirmation

## Architecture

```
sprinto/
├── app.py                        # Streamlit entrypoint
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── schema.sql                    # Supabase DDL (11 tables + pgvector)
├── config/
│   └── default_extraction.json   # Default extraction field config
├── services/
│   ├── database.py               # Supabase REST + psycopg2 CRUD
│   ├── parser.py                 # PDF/DOCX parsing + section-aware chunking
│   ├── embeddings.py             # HuggingFace embedding generation
│   ├── ai_engine.py              # Groq LLM: extraction, scoring, questions
│   ├── rag.py                    # Hybrid retrieval (semantic + keyword + RRF)
│   ├── duplicate.py              # Exact + fuzzy duplicate detection
│   └── utils.py                  # Helpers (hashing, text utils)
├── pages/
│   ├── 1_Dashboard.py            # KPIs and recent activity
│   ├── 2_Upload_Resumes.py       # Multi-file upload pipeline
│   ├── 3_Role_Management.py      # JD CRUD + scan triggering
│   ├── 4_Candidate_Review.py     # Ranking, filtering, feedback
│   ├── 5_Candidate_Detail.py     # Profile, scores, evidence, interview prep
│   ├── 6_Multi_Role_Match.py     # Cross-role comparison
│   ├── 7_Scan_History.py         # Audit trail
│   └── 8_Settings.py             # Config editor + batch re-parse
└── assets/
    └── style.css                 # Custom styling
```

### AI and Prompting Strategy

| Task | Provider | Model | Strategy |
|------|----------|-------|----------|
| Field extraction | Groq | Llama 3.3 70B | Structured JSON prompt with field definitions |
| Scoring | Groq | Llama 3.3 70B | Evidence-backed prompt with retrieved chunks + full resume |
| Quality check | Groq | Llama 3.3 70B | Completeness checklist + quality rubric |
| Interview questions | Groq | Llama 3.3 70B | Gap-targeted questions from scoring weak areas |
| Embeddings | HuggingFace | all-MiniLM-L6-v2 | 384-dim, via Inference API with caching |

### RAG Pipeline

1. **Parse**: Extract text from PDF/DOCX
2. **Chunk**: Section-aware sliding window (500 words, 100 overlap)
3. **Embed**: HuggingFace all-MiniLM-L6-v2 for each chunk
4. **Store**: pgvector in Supabase PostgreSQL
5. **Retrieve**: Hybrid search (cosine similarity + keyword matching)
6. **Merge**: Reciprocal Rank Fusion (RRF)
7. **Score**: Groq LLM uses retrieved evidence + full resume to score

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [Supabase](https://supabase.com) project (free tier works)
- [Groq](https://console.groq.com) API key (free tier available)
- [HuggingFace](https://huggingface.co/settings/tokens) access token

### 2. Setup Supabase

1. Create a Supabase project
2. Go to **SQL Editor** and run the contents of `schema.sql`
3. This creates all 11 tables and enables the `vector` extension

### 3. Configure Environment

```bash
cp .env.example .env
```

Fill in your credentials:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
SUPABASE_DB_URL=postgresql://postgres:password@db.xxxx.supabase.co:5432/postgres
GROQ_API_KEY=your-groq-api-key
HF_TOKEN=your-huggingface-token
```

Use the Supabase service role key (not the anon key) for full table access.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Usage

1. **Upload Resumes**: Go to Upload, drop PDF/DOCX files, process them
2. **Create a Role**: Go to Role Management, paste a JD, add weighted requirements
3. **Scan Candidates**: Click "Scan All" on a role to score all resumes
4. **Review Results**: Go to Candidate Review, filter/sort, provide feedback
5. **Deep Dive**: Click Detail for evidence, requirement breakdown, interview questions
6. **Multi-Role**: Go to Multi-Role Match to compare one candidate across roles
7. **Adjust and Re-scan**: Update config in Settings, batch re-parse, then re-scan

## Edge Cases Handled

- **Corrupted/scanned PDFs**: PyPDF2 to pdfplumber fallback + error status
- **Empty resumes**: Minimum text length check (20 chars)
- **Unsupported formats**: Clear error for non-PDF/DOCX files
- **Missing fields**: Null values handled gracefully in extraction
- **API failures**: Zero-vector fallback for failed embeddings
- **Duplicate uploads**: Both exact hash and semantic similarity detection
- **Long resumes**: Text truncation for API limits
- **Failed processing**: Resume status set to "error" instead of stuck "parsing"

## Deployment

### Streamlit Community Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create a new app pointing to `app.py`
4. Add secrets (equivalent to `.env`) in the Streamlit dashboard

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```
