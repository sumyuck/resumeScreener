-- Resume Screener: Supabase Schema
-- Run in Supabase SQL Editor to create all tables.

CREATE EXTENSION IF NOT EXISTS vector;

-- Users (recruiters)
CREATE TABLE IF NOT EXISTS users (
    id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now()
);

-- Roles / Job Descriptions
CREATE TABLE IF NOT EXISTS roles (
    id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    title         TEXT NOT NULL,
    department    TEXT,
    jd_text       TEXT NOT NULL,
    requirements  JSONB DEFAULT '[]'::jsonb,
    status        TEXT DEFAULT 'active' CHECK (status IN ('active', 'archived')),
    created_by    UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now()
);

-- Resumes
CREATE TABLE IF NOT EXISTS resumes (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    filename        TEXT NOT NULL,
    file_type       TEXT NOT NULL CHECK (file_type IN ('pdf', 'docx')),
    file_hash       TEXT NOT NULL,
    raw_text        TEXT,
    candidate_name  TEXT,
    candidate_email TEXT,
    text_hash       TEXT,
    status          TEXT DEFAULT 'parsed' CHECK (status IN ('uploaded', 'parsing', 'parsed', 'error')),
    quality_score   REAL,
    quality_notes   TEXT,
    uploaded_by     UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_resumes_file_hash ON resumes(file_hash);
CREATE INDEX IF NOT EXISTS idx_resumes_text_hash ON resumes(text_hash);

-- Extraction Configs
CREATE TABLE IF NOT EXISTS extraction_configs (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name        TEXT NOT NULL,
    fields      JSONB NOT NULL,
    is_default  BOOLEAN DEFAULT false,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Extracted Fields (per resume)
CREATE TABLE IF NOT EXISTS extracted_fields (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    resume_id   UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    config_id   UUID REFERENCES extraction_configs(id) ON DELETE SET NULL,
    fields      JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_extracted_fields_resume ON extracted_fields(resume_id);

-- Resume Chunks
CREATE TABLE IF NOT EXISTS resume_chunks (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    resume_id   UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    section     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_resume ON resume_chunks(resume_id);

-- Chunk Embeddings (pgvector)
-- Using 384 dimensions for sentence-transformers/all-MiniLM-L6-v2
-- If migrating from 768-dim embeddings, run: ALTER TABLE chunk_embeddings ALTER COLUMN embedding TYPE vector(384);
-- Then re-embed all resumes via Settings > Batch Re-Parse.
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    chunk_id    UUID NOT NULL REFERENCES resume_chunks(id) ON DELETE CASCADE,
    resume_id   UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    embedding   vector(384) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_resume ON chunk_embeddings(resume_id);

-- Scan Results (per resume x role)
CREATE TABLE IF NOT EXISTS scan_results (
    id                UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    resume_id         UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    role_id           UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    score             REAL NOT NULL CHECK (score >= 0 AND score <= 10),
    summary           TEXT,
    evidence          JSONB DEFAULT '[]'::jsonb,
    requirement_scores JSONB DEFAULT '[]'::jsonb,
    confidence        TEXT DEFAULT 'medium' CHECK (confidence IN ('high', 'medium', 'low')),
    flagged_for_review BOOLEAN DEFAULT false,
    config_snapshot   JSONB,
    scan_history_id   UUID,
    created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_scan_resume ON scan_results(resume_id);
CREATE INDEX IF NOT EXISTS idx_scan_role ON scan_results(role_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_scan_resume_role ON scan_results(resume_id, role_id);

-- Duplicate Flags
CREATE TABLE IF NOT EXISTS duplicate_flags (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    resume_id       UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    duplicate_of    UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    role_id         UUID REFERENCES roles(id) ON DELETE CASCADE,
    flag_type       TEXT NOT NULL CHECK (flag_type IN ('exact', 'possible')),
    similarity      REAL,
    resolved        BOOLEAN DEFAULT false,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dup_resume ON duplicate_flags(resume_id);

-- Recruiter Feedback
CREATE TABLE IF NOT EXISTS recruiter_feedback (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    resume_id   UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    role_id     UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    decision    TEXT NOT NULL CHECK (decision IN ('shortlist', 'reject', 'maybe')),
    notes       TEXT,
    decided_by  UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_resume_role ON recruiter_feedback(resume_id, role_id);

-- Scan History (audit log)
CREATE TABLE IF NOT EXISTS scan_history (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    role_id         UUID REFERENCES roles(id) ON DELETE CASCADE,
    resume_count    INTEGER DEFAULT 0,
    config_id       UUID REFERENCES extraction_configs(id) ON DELETE SET NULL,
    scan_type       TEXT DEFAULT 'manual' CHECK (scan_type IN ('manual', 'batch_rescan', 'auto')),
    status          TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    started_at      TIMESTAMPTZ DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    triggered_by    UUID REFERENCES users(id) ON DELETE SET NULL,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_scan_hist_role ON scan_history(role_id);

-- Auto-update updated_at triggers
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_updated BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_roles_updated BEFORE UPDATE ON roles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_resumes_updated BEFORE UPDATE ON resumes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_feedback_updated BEFORE UPDATE ON recruiter_feedback
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
