"""
Resume parsing: PDF and DOCX text extraction with section-aware chunking.
"""

import re
import io
from typing import BinaryIO

import PyPDF2
import pdfplumber
from docx import Document


def parse_pdf(file_bytes: bytes) -> str:
    text = _parse_pdf_pypdf2(file_bytes)
    if not text or len(text.strip()) < 50:
        text = _parse_pdf_pdfplumber(file_bytes)
    return text.strip() if text else ""


def _parse_pdf_pypdf2(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n\n".join(pages)
    except Exception:
        return ""


def _parse_pdf_pdfplumber(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
            return "\n\n".join(pages)
    except Exception:
        return ""


def parse_docx(file_bytes: bytes) -> str:
    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        return "\n".join(paragraphs)
    except Exception:
        return ""


def parse_resume(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    if ext == 'pdf':
        text = parse_pdf(file_bytes)
    elif ext in ('docx', 'doc'):
        text = parse_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: .{ext}. Only PDF and DOCX are supported.")

    if not text or len(text.strip()) < 20:
        raise ValueError(
            f"Could not extract meaningful text from '{filename}'. "
            "The file may be scanned/image-based or corrupted."
        )
    return text


SECTION_PATTERNS = [
    (r'(?i)\b(work\s*experience|professional\s*experience|employment\s*history|experience)\b', 'experience'),
    (r'(?i)\b(education|academic|qualifications)\b', 'education'),
    (r'(?i)\b(skills|technical\s*skills|core\s*competencies|technologies)\b', 'skills'),
    (r'(?i)\b(projects|personal\s*projects|key\s*projects)\b', 'projects'),
    (r'(?i)\b(certifications?|licenses?|credentials)\b', 'certifications'),
    (r'(?i)\b(summary|objective|profile|about\s*me)\b', 'summary'),
    (r'(?i)\b(awards?|achievements?|honors?)\b', 'awards'),
    (r'(?i)\b(publications?|research|papers?)\b', 'publications'),
    (r'(?i)\b(volunteer|community|extracurricular)\b', 'volunteer'),
    (r'(?i)\b(languages?|interests?|hobbies)\b', 'other'),
]


def detect_section(text: str) -> str | None:
    first_line = text.split('\n')[0].strip()
    for pattern, section in SECTION_PATTERNS:
        if re.search(pattern, first_line):
            return section
    return None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    lines = text.split('\n')
    sections = []
    current_section = "general"
    current_lines = []

    for line in lines:
        detected = detect_section(line)
        if detected and current_lines:
            sections.append((current_section, '\n'.join(current_lines)))
            current_section = detected
            current_lines = [line]
        else:
            if detected:
                current_section = detected
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, '\n'.join(current_lines)))

    chunks = []
    chunk_index = 0

    for section_name, section_text in sections:
        words = section_text.split()
        if len(words) <= chunk_size:
            if section_text.strip():
                chunks.append({
                    "chunk_index": chunk_index,
                    "chunk_text": section_text.strip(),
                    "section": section_name,
                })
                chunk_index += 1
        else:
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_words = words[start:end]
                chunk_t = ' '.join(chunk_words).strip()
                if chunk_t:
                    chunks.append({
                        "chunk_index": chunk_index,
                        "chunk_text": chunk_t,
                        "section": section_name,
                    })
                    chunk_index += 1
                start += chunk_size - overlap

    if not chunks and text.strip():
        chunks.append({
            "chunk_index": 0,
            "chunk_text": text.strip()[:2000],
            "section": "general",
        })

    return chunks
