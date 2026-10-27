"""Settings for the annual-report comparison + Q&A system. Everything tunable lives here or in .env."""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional: read API keys from a .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT / "data" / "reports"))  # put the annual-report PDFs here
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))        # written by ingest.py
METRICS_CSV = Path(os.getenv("METRICS_CSV", ROOT / "data" / "metrics.csv"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))

# Company -> lower-case aliases. A PDF belongs to a company when its file name contains an alias
# (or when it is listed in REPORTS below). Questions are routed to companies the same way.
COMPANIES = {
    "HCLTech": ["hcltech", "hcl tech", "hcl technologies", "hcl"],
    "Infosys": ["infosys", "infy"],
    "TCS": ["tcs", "tata consultancy services", "tata consultancy"],
}

# Optional source register: PDF file name -> details used in citations and on the Sources tab.
REPORTS: dict[str, dict] = {
    # "TCS_Annual_Report_2025-26.pdf": {"company": "TCS", "title": "TCS Integrated Annual Report 2025-26",
    #     "fiscal_year": "FY2025-26", "url": "https://...", "downloaded": "2026-10-27"},
}

# NSE-listed IT firms that are NOT in the corpus. A question naming only these is refused before retrieval.
OUT_OF_CORPUS = [
    "wipro", "tech mahindra", "ltimindtree", "lti mindtree", "mphasis", "coforge", "persistent systems",
    "hexaware", "l&t technology services", "ltts", "cyient", "zensar", "birlasoft", "kpit", "sonata software",
    "oracle financial services", "cognizant", "accenture", "capgemini",
]

# Retrieval
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
PRELOAD_RERANKER = os.getenv("PRELOAD_RERANKER", "1") == "1"  # load at startup so the retry isn't slowed by it
CHUNK_CHARS = 1200         # chunks never cross a page boundary, so every chunk has exactly one page number
CHUNK_OVERLAP = 200
TOP_K = 6                  # excerpts sent to the LLM when the question is about one company
TOP_K_PER_COMPANY = 3      # ...per company when the question spans several companies (or names none)
RETRY_K = 8                # excerpts after reranking, on the single retry
CANDIDATE_POOL = 40        # hybrid candidates the reranker chooses from
MIN_TERM_COVERAGE = 0.3    # evidence gate: share of the question's key terms that must appear in retrieved text
RETRY_ON_NOT_FOUND = True  # also retry when the LLM says "not found" (better recall, slower refusals)
# Ingestion / parser
TABLE_MIN_DIGIT_SHARE = 0.10   # pages with at least this share of digits are read row by row and searched for tables
MARGIN_SHARE = 0.08            # top/bottom share of a page treated as header/footer margin
BOILERPLATE_MIN_SHARE = 0.3    # a margin line repeated on this share of a report's pages is dropped from chunks
FAISS_THREADS = 1              # exact search over a few thousand vectors; extra threads only add overhead

# LLM (see .env.example). Empty provider = auto-detect from whichever API key is set; "none" = extractive mode.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
MAX_ANSWER_TOKENS = 800

# R1 chart: columns of data/metrics.csv and the metrics extract_metrics.py looks for in each report.
METRIC_FIELDS = ["company", "metric", "value", "value_text", "unit", "fiscal_year", "report", "file", "pdf_page",
                 "page_label", "quote", "note", "auto_check", "human_check"]
METRICS = {
    "revenue": {
        "label": "Revenue from operations (consolidated)",
        "query": "consolidated revenue from operations for the year crore",
        "hint": "consolidated revenue from operations for the latest fiscal year (prefer ₹ crore)",
    },
    "net_profit": {
        "label": "Net profit (consolidated)",
        "query": "consolidated net profit profit for the year attributable to shareholders crore",
        "hint": "consolidated net profit / profit for the year attributable to shareholders (prefer ₹ crore)",
    },
    "operating_margin": {
        "label": "Operating margin",
        "query": "operating margin EBIT margin percentage for the year",
        "hint": "operating (EBIT) margin in %; say in the note if the report calls it EBIT margin",
    },
    "headcount": {
        "label": "Employees at year end",
        "query": "total number of employees headcount workforce as at March 31",
        "hint": "total employees at fiscal year end, whole company",
    },
    "attrition": {
        "label": "Attrition rate",
        "query": "attrition rate employee turnover LTM voluntary",
        "hint": "attrition / employee turnover rate in %; record its definition (LTM, voluntary, IT services, BRSR)",
    },
}
