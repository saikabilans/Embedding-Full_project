from pathlib import Path
from datetime import datetime
import hashlib
import json
import re

import fitz
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"

OUTPUT_DIR = BASE_DIR / "output" / "task2"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for folder in [
    DOCS_DIR,
    OUTPUT_DIR,
    MODELS_DIR,
    REPORTS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


def create_sample_documents():

    if len(list(DOCS_DIR.iterdir())) >= 5:
        return

    policy_pdf = DOCS_DIR / "company_policy.pdf"

    pdf = fitz.open()
    page = pdf.new_page()

    text = """
Company Policy

Working Hours

The standard working hours are from 10 AM to 6 PM.
Employees should login before the beginning of their workday.

Leave Policy

Employees should submit leave requests through the HRMS portal.
Leave requests should be submitted before taking planned leave.

Break Policy

Employees are provided tea breaks and a lunch break according
to company policy.
"""

    page.insert_text((50, 50), text)
    pdf.save(policy_pdf)
    pdf.close()

    manual_pdf = DOCS_DIR / "employee_manual.pdf"

    pdf = fitz.open()
    page = pdf.new_page()

    text = """
Employee Manual

Attendance

Employees must maintain regular attendance.
Employees should login and logout correctly in the HRMS system.

Overtime

Work performed beyond normal working hours may be treated
as overtime according to company policy.

Salary

Salary information is available through the employee portal.
"""

    page.insert_text((50, 50), text)
    pdf.save(manual_pdf)
    pdf.close()

    docx_file = DOCS_DIR / "training_manual.docx"

    doc = Document()

    doc.add_heading(
        "Training Manual",
        level=1
    )

    doc.add_heading(
        "Machine Learning",
        level=2
    )

    doc.add_paragraph(
        "Machine learning allows computers to learn patterns "
        "from data and make predictions."
    )

    doc.add_heading(
        "Embeddings",
        level=2
    )

    doc.add_paragraph(
        "Embeddings convert text into numerical vectors that "
        "can be compared using similarity measures."
    )

    doc.save(docx_file)

    notes_file = DOCS_DIR / "project_notes.txt"

    notes_file.write_text(
        """
Project Notes

Semantic Search

Semantic search finds documents based on meaning rather
than only exact keyword matching.

Vector Database

A vector database stores embeddings and allows similarity
search over large collections of vectors.
""",
        encoding="utf-8"
    )

    guide_file = DOCS_DIR / "ai_guide.txt"

    guide_file.write_text(
        """
AI Guide

Artificial Intelligence

Artificial intelligence enables computers to perform tasks
that normally require human intelligence.

Natural Language Processing

Natural language processing allows systems to process and
understand human language.

Retrieval

Retrieval systems find relevant information from a collection
of documents.
""",
        encoding="utf-8"
    )


def extract_pdf(path):

    documents = []

    pdf = fitz.open(path)

    for page_number, page in enumerate(pdf, start=1):

        documents.append({
            "text": page.get_text(),
            "page": page_number,
            "section": "PDF Content"
        })

    pdf.close()

    return documents


def extract_docx(path):

    doc = Document(path)

    text = "\n".join(
        paragraph.text
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    )

    return [{
        "text": text,
        "page": 1,
        "section": "DOCX Content"
    }]


def extract_txt(path):

    text = path.read_text(
        encoding="utf-8"
    )

    return [{
        "text": text,
        "page": 1,
        "section": "TXT Content"
    }]


def extract_document(path):

    extension = path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(path)

    if extension == ".docx":
        return extract_docx(path)

    if extension == ".txt":
        return extract_txt(path)

    return []


def fixed_token_chunking(
    text,
    chunk_size=512,
    overlap=50
):

    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for start in range(
        0,
        len(words),
        step
    ):

        chunk = " ".join(
            words[start:start + chunk_size]
        )

        if chunk:
            chunks.append(chunk)

    return chunks


def sentence_chunking(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def recursive_chunking(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    return splitter.split_text(text)


def stable_chunk_id(
    source,
    page,
    chunk_index,
    text
):

    value = (
        f"{source}|{page}|"
        f"{chunk_index}|{text}"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def describe_chunk(chunk):

    return (
        f"{chunk['source']} | "
        f"page {chunk['page']} | "
        f"{chunk['section']} | "
        f"chunk {chunk['chunk_index']}"
    )


create_sample_documents()

all_chunks = []

fixed_count = 0
sentence_count = 0
recursive_count = 0

for document_path in DOCS_DIR.iterdir():

    if not document_path.is_file():
        continue

    pages = extract_document(
        document_path
    )

    for page_data in pages:

        text = page_data["text"]

        fixed_chunks = fixed_token_chunking(
            text
        )

        sentence_chunks = sentence_chunking(
            text
        )

        recursive_chunks = recursive_chunking(
            text
        )

        fixed_count += len(fixed_chunks)
        sentence_count += len(sentence_chunks)
        recursive_count += len(recursive_chunks)

        for index, chunk_text in enumerate(
            recursive_chunks
        ):

            chunk = {
                "id": stable_chunk_id(
                    document_path.name,
                    page_data["page"],
                    index,
                    chunk_text
                ),
                "source": document_path.name,
                "page": page_data["page"],
                "section": page_data["section"],
                "created_date": datetime.now().isoformat(),
                "chunk_index": index,
                "doc_type": document_path.suffix.replace(
                    ".",
                    ""
                ),
                "text": chunk_text
            }

            chunk["citation"] = describe_chunk(
                chunk
            )

            all_chunks.append(chunk)


chunks_file = OUTPUT_DIR / "chunks.json"

chunks_file.write_text(
    json.dumps(
        all_chunks,
        indent=2
    ),
    encoding="utf-8"
)

# Also save model-ready copy
model_chunks_file = MODELS_DIR / "chunks.json"

model_chunks_file.write_text(
    json.dumps(
        all_chunks,
        indent=2
    ),
    encoding="utf-8"
)


original_ids = {
    chunk["id"]
    for chunk in all_chunks
}

modified_chunks = []

for chunk in all_chunks:

    modified = chunk.copy()

    if modified_chunks == []:
        modified["text"] += " Modified paragraph."

    modified["id"] = stable_chunk_id(
        modified["source"],
        modified["page"],
        modified["chunk_index"],
        modified["text"]
    )

    modified_chunks.append(modified)

modified_ids = {
    chunk["id"]
    for chunk in modified_chunks
}

changed_ids = original_ids.symmetric_difference(
    modified_ids
)


report = f"""


Documents:
{len(list(DOCS_DIR.iterdir()))}

Fixed token chunks:
{fixed_count}

Sentence chunks:
{sentence_count}

RecursiveCharacterTextSplitter chunks:
{recursive_count}

Recursive configuration:
chunk_size = 512
chunk_overlap = 50

Total stored chunks:
{len(all_chunks)}

Stable IDs:
SHA256 hash

Original unique IDs:
{len(original_ids)}

Changed IDs after one paragraph edit:
{len(changed_ids)}

Metadata fields:
source
page
section
created_date
chunk_index
doc_type
id
citation

Sample citation:
{all_chunks[0]["citation"] if all_chunks else "No chunks"}
"""

report_file = REPORTS_DIR / "task2_report.txt"

report_file.write_text(
    report.strip(),
    encoding="utf-8"
)