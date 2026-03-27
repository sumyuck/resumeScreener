# AI Resume Screener

**Live App:** https://resumescreener123.streamlit.app/

Recruiters often spend significant time manually reviewing unstructured resumes, checking fit against job requirements, identifying possible duplicates, and preparing first-round screening questions.

Thus, A resume screening tool that uses AI to score candidates against job descriptions. It parses resumes, pulls out structured data, scores each candidate using a hybrid retrieval + LLM pipeline, and gives evidence-backed justifications for every score.

---

## Screenshots

### Dashboard
![Dashboard](assets/screenshots/dashboard.png)

### Job Management
![Jobs](assets/screenshots/jobs.png)

### Resume Upload
![Resumes](assets/screenshots/resumes.png)

### Screening Results
![Screening](assets/screenshots/screening.png)

### Job Fit Breakdown
![Job Fit](assets/screenshots/job-fit.png)

### Resume Evidence
![Resume Evidence](assets/screenshots/resume-evidence.png)

### Phone Screen Prep
![Phone Prep](assets/screenshots/phone-prep.png)

### Multi-Role Match
![Multi-Role Match](assets/screenshots/multi-role-match.png)

### Custom Extraction Config
![Custom Config](assets/screenshots/custom-config.png)

---

## Architecture

![Architecture Diagram](assets/screenshots/architecture_diagram.png)

### Project Structure

```
sprinto/
├── app.py                        # Streamlit entrypoint
├── schema.sql                    # Full database DDL + pgvector setup
├── config/
│   └── default_extraction.json   # Default extraction fields
├── services/
│   ├── ai_engine.py              # LLM scoring, extraction, guardrails
│   ├── rag.py                    # Hybrid retrieval (semantic + keyword)
│   ├── parser.py                 # PDF/DOCX parsing + section-aware chunking
│   ├── embeddings.py             # HuggingFace embedding generation
│   ├── duplicate.py              # Multi-layer duplicate detection
│   ├── database.py               # Supabase CRUD operations
│   └── utils.py                  # Helpers
├── pages/
│   ├── 1_Dashboard.py            # Overview + quick actions
│   ├── 2_Jobs.py                 # JD management + scan triggering
│   ├── 3_Resumes.py              # Upload pipeline + duplicate blocking
│   ├── 4_Screening.py            # Candidate ranking + feedback
│   ├── 5_Candidate_Detail.py     # Per-candidate deep dive
│   ├── 6_Multi_Role_Match.py     # Cross-role / cross-candidate comparison
│   └── 7_Settings.py             # Config editor + batch re-parse
└── assets/
    └── style.css
```

---

## How Scoring Works

The scoring system has two stages, the LLM handles contextual understanding, and deterministic code handles the math.

### Step 1: Resume Ingestion

When a resume is uploaded:
1. **Parse**: Extract text from PDF (with a fallback parser if the first one fails) or DOCX
2. **Chunk**: Split text into chunks, respecting section boundaries (Experience, Skills, Education, etc.) so context stays intact. Each chunk gets a section label like `[Experience]` prepended
3. **Embed**: Generate 384-dim vector embeddings using HuggingFace's `all-MiniLM-L6-v2`
4. **Store**: Save chunks and embeddings in Supabase PostgreSQL with pgvector

### Step 2: Retrieval (RAG)

When scoring against a job, the system retrieves the most relevant resume evidence:

- **Requirement-centric search**: Each requirement from the JD gets its own embedding-based search, so we gather evidence for every requirement, not just the most prominent ones
- **Keyword matching**: Extracts tech skills from the JD and matches against chunks. Chunks from experience/project sections get a slight relevance boost
- **Reciprocal Rank Fusion**: Merges the semantic and keyword results into a single ranked list, keeping chunks that showed up in both methods

### Step 3: Hybrid Scoring

The LLM doesn't assign numeric scores directly, that was too unstable across calls. Instead:

1. **LLM classifies each requirement** as `strong_match`, `moderate_match`, `weak_match`, or `no_match` based on the retrieved evidence
2. **Deterministic code maps those to numbers**:
   - strong = 9, moderate = 6.5, weak = 4, no match = 1.5
3. **Weighted by category**: must_have requirements carry 3x weight, good_to_have 2x, bonus 1x
4. **Final score** = weighted average, normalized to 0-10

The LLM is prompted to be fair but slightly generous, it treats equivalent technologies (e.g., PostgreSQL for a SQL requirement) as strong matches and gives credit for transferable experience.

---

## Feature Breakdown

### Resume Parsing
- Tries PyPDF2 first, falls back to pdfplumber if text extraction is too sparse
- DOCX support via python-docx
- Section-aware chunking - recognizes headers like "Work Experience", "Education", "Skills" etc. via regex and keeps chunks within their sections
- Detects garbled text (corrupted PDFs, image-only scans) and shows quality warnings during upload

### Field Extraction
- Uses Groq's Llama 3.3 70B to extract structured fields (name, email, skills, experience, etc.) from resume text
- Fields are fully configurable through the Settings UI - add, remove, or edit fields and re-parse all resumes

### Duplicate Detection
Three layers, each catching a different scenario:

- **File hash**: SHA-256 of raw file bytes → catches exact re-uploads
- **Text hash**: SHA-256 of normalized text (lowercased, whitespace collapsed, punctuation stripped) → catches same content in different formats (PDF vs DOCX)
- **Fuzzy embedding**: Cosine similarity at 0.95 threshold → catches near-identical resumes with minor edits

Exact duplicates are blocked before the resume even enters the database, with a message like "Skipped — duplicate of John Doe". Fuzzy matches are flagged as warnings.

### Multi-Role Matching
Two modes:
- **Candidate → Roles**: Score one person against multiple jobs side-by-side
- **Role → Candidates**: Compare multiple candidates for one role in a ranked table

## Other Features
- Custom config for parsing
- Batch re-parsing
- Generates practical call sheets for HR
- JD Validation
- Prompt Injection Defense

---

## Edge Cases Handled

| Edge Case | What Happens |
|-----------|-------------|
| Corrupted / scanned PDF | Fallback parser kicks in; garbled text detected and warned |
| Empty or image-only resume | Blocked with clear error if < 20 chars extracted |
| Unsupported file format | Only PDF/DOCX accepted |
| Prompt injection in resume | Patterns stripped, resume auto-flagged for review |
| Incomplete job description | Blocked if too short or missing requirements |
| Same file re-uploaded | Blocked pre-upload via file hash |
| Same resume, different format | Caught via normalized text hash |
| Near-identical resumes | Flagged via embedding similarity |
| Same-name candidates | Dropdown labels include filename for disambiguation |
| API failures | Zero-vector fallback for embeddings; error status for failed parses |
| Very long resumes | Text truncated to stay within API limits |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Database | Supabase PostgreSQL + pgvector |
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | HuggingFace Inference API (all-MiniLM-L6-v2, 384-dim) |
| PDF Parsing | PyPDF2 + pdfplumber (dual fallback) |
| DOCX Parsing | python-docx |

---

## Setup

### Prerequisites
- Python 3.10+
- [Supabase](https://supabase.com) project (free tier works)
- [Groq](https://console.groq.com) API key
- [HuggingFace](https://huggingface.co/settings/tokens) access token

### 1. Database

Create a Supabase project, go to SQL Editor, and run `schema.sql`. This sets up all the tables and enables pgvector.

### 2. Environment

.env

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
SUPABASE_DB_URL=postgresql://postgres:password@db.xxxx.supabase.co:5432/postgres
GROQ_API_KEY=your-groq-api-key
HF_TOKEN=your-huggingface-token
```

Use the service role key (not the anon key) from Project Settings → API.

### 3. Install & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Create app at [share.streamlit.io](https://share.streamlit.io) pointing to `app.py`
3. Add secrets in the Streamlit dashboard (same as `.env` variables)
