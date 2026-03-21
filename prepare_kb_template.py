import os
import re
import json
import shutil
import hashlib
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import chromadb
import litellm
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from pypdf import PdfReader, PdfWriter

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.litellm import LiteLLM
from llama_index.vector_stores.chroma import ChromaVectorStore

try:
    from utils.cortex_litellm_provider import register_cortex_provider
except Exception:
    def register_cortex_provider():
        return None

# =============================================================================
# Profiles
# =============================================================================

PROFILE_CONFIGS = {
    "policy_guidance": {
        "description": "Policy, risk triage, and validation guidance KB",
        "default_pdf_dir": "./PDFdata_policy",
        "collection_name": "policy_guidance_collection",
    },
    "model_evidence": {
        "description": "Model documentation and model evidence KB",
        "default_pdf_dir": "./PDFdata_model",
        "collection_name": "model_evidence_collection",
    },
}
MANIFEST_SCHEMA_VERSION = "3.0"

# =============================================================================
# Environment Config
# =============================================================================

CORTEX_LLM_MODEL = os.getenv("CORTEX_LLM_MODEL", "cortex/vertex_ai/gemini-2.5-flash")
CORTEX_EMBED_MODEL = os.getenv("CORTEX_EMBED_MODEL", "cortex/vertex_ai/text-embedding-004")
CORTEX_CUSTOM_LLM_PROVIDER = os.getenv("CORTEX_CUSTOM_LLM_PROVIDER", "cortex")
CORTEX_LLM_TEMPERATURE = float(os.getenv("CORTEX_LLM_TEMPERATURE", "0.1"))
CORTEX_LLM_MAX_TOKENS = int(os.getenv("CORTEX_LLM_MAX_TOKENS", "1024"))
CORTEX_CONTEXT_WINDOW = int(os.getenv("CORTEX_CONTEXT_WINDOW", "32768"))

DOCLING_ARTIFACTS_PATH = os.getenv("DOCLING_ARTIFACTS_PATH", "/home/jupyter/docling-models")
DOCLING_ENFORCE_LOCAL_MODELS = os.getenv("DOCLING_ENFORCE_LOCAL_MODELS", "1").lower() not in {"0", "false", "no"}
DOCLING_ENABLE_OCR = os.getenv("DOCLING_ENABLE_OCR", "0").lower() in {"1", "true", "yes"}
DOCLING_ENABLE_TABLE_STRUCTURE = os.getenv("DOCLING_ENABLE_TABLE_STRUCTURE", "0").lower() in {"1", "true", "yes"}
RAPIDOCR_MODELS_PATH = os.getenv("RAPIDOCR_MODELS_PATH", "")
RAPIDOCR_DET_MODEL_PATH = os.getenv("RAPIDOCR_DET_MODEL_PATH", "")
RAPIDOCR_REC_MODEL_PATH = os.getenv("RAPIDOCR_REC_MODEL_PATH", "")
RAPIDOCR_CLS_MODEL_PATH = os.getenv("RAPIDOCR_CLS_MODEL_PATH", "")

BASE_CHUNK_SIZE = 900
BASE_CHUNK_OVERLAP = 120
WINDOW_SIZE = 3
VECTOR_TOP_K = 4
BM25_TOP_K = 4
MIN_MEANINGFUL_TEXT_LEN = 80
MAX_SECTION_DOC_CHARS = 3200
DELETE_TEMP_PAGE_PDFS = False

VERIFICATION_QUERIES = {
    "policy_guidance": {
        "vector": "Summarize the main governance or validation guidance in this corpus.",
        "bm25": "validation scope risk triage controls governance"
    },
    "model_evidence": {
        "vector": "What does this model do and how is it described?",
        "bm25": "inputs outputs assumptions limitations monitoring"
    },
}

# =============================================================================
# Helpers
# =============================================================================

def log(message: str):
    print(f"[prepare_kb] {message}", flush=True)


def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def file_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_stem(path: str) -> str:
    return Path(path).stem.replace(" ", "_")


def english_sentence_splitter(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def short_preview(text: str, length: int = 220) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:length] + ("..." if len(text) > length else "")


def normalize_text_for_display(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def clean_title_candidate(text: str) -> str:
    value = normalize_text_for_display(text)
    if not value:
        return ""
    bad_patterns = [
        r"^<!--.*?-->$", r"^\[image\]$", r"^\[figure\]$", r"^\[table\]$",
        r"^image$", r"^figure$", r"^table$", r"^page \d+$", r"^[\W_]+$"
    ]
    for pattern in bad_patterns:
        if re.match(pattern, value, flags=re.I):
            return ""
    if len(value) < 3:
        return ""
    if value.count("|") > 6:
        return ""
    return value[:160]


def parse_version(text: str) -> Optional[str]:
    patterns = [
        r"\b(?:version|ver|v)[\s._-]*([0-9]+(?:\.[0-9]+){0,2})\b",
        r"\b(ai rmf\s*1\.0)\b",
        r"\b([A-Z]{2,10}\s+\d{1,4}[.-]\d{1,3})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return normalize_text_for_display(match.group(1))
    return None


def parse_effective_date(text: str) -> Optional[str]:
    patterns = [
        r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b",
        r"\b(\d{1,2}[-/]\d{1,2}[-/](?:20)?\d{2})\b",
        r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s+20\d{2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1)
    return None


def infer_document_type(text: str, profile: str) -> str:
    lowered = (text or "").lower()
    if profile == "policy_guidance":
        mapping = [
            ("policy", ["policy", "policy standard"]),
            ("standard", ["standard", "principle", "framework"]),
            ("procedure", ["procedure", "workflow", "process"]),
            ("guidance", ["guidance", "playbook", "manual"]),
            ("methodology", ["methodology", "approach", "validation approach"]),
            ("appendix", ["appendix", "annex"]),
        ]
    else:
        mapping = [
            ("model_document", ["model document", "model overview", "model specification", "model development"]),
            ("implementation_note", ["implementation", "technical design", "solution design"]),
            ("monitoring_report", ["monitoring", "performance report", "monitoring report"]),
            ("validation_report", ["validation report", "independent review"]),
            ("methodology", ["methodology", "approach"]),
            ("appendix", ["appendix", "annex"]),
        ]
    for doc_type, markers in mapping:
        if any(marker in lowered for marker in markers):
            return doc_type
    return "unknown"


def infer_business_area(source_file: str, title_text: str, leading_text: str, profile: str) -> Optional[str]:
    search_text = " ".join([
        source_file.replace("_", " ").replace("-", " "),
        title_text or "",
        (leading_text or "")[:500],
    ]).lower()

    if profile == "model_evidence":
        areas = {
            "credit_risk": ["credit risk", "pd model", "lgd model", "ead model", "scorecard", "loan underwriting"],
            "market_risk": ["market risk", "var model", "stressed var", "trading book"],
            "fraud": ["fraud detection", "fraud model", "fraud analytics"],
            "compliance": ["aml", "kyc", "compliance monitoring", "sanctions screening"],
            "finance": ["ifrs 9", "cecl", "finance forecasting", "accounting model"],
            "retail_banking": ["retail banking", "consumer lending", "mortgage model"],
        }
    else:
        areas = {
            "ai_governance": ["ai risk management", "ai governance", "ai rmf", "responsible ai"],
            "model_risk": ["model risk", "validation", "independent validation", "model governance"],
            "enterprise_risk": ["risk management", "governance framework", "controls framework"],
        }

    for area, markers in areas.items():
        if any(marker in search_text for marker in markers):
            return area
    return None


def infer_model_name(text: str) -> Optional[str]:
    patterns = [
        r"\bmodel[\s:_-]+([a-z0-9][a-z0-9 _/\-]{2,60})",
        r"\b(pd|lgd|ead|scorecard|rating model|forecast model)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return normalize_text_for_display(match.group(1 if match.lastindex else 0)).strip(" -_:")
    return None


def infer_owner(text: str) -> Optional[str]:
    match = re.search(r"\bowner[\s:]+([A-Za-z0-9 &/_-]{3,80})", text, flags=re.I)
    return normalize_text_for_display(match.group(1)) if match else None


def infer_approval_status(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    for status in ["approved", "draft", "final", "pending", "obsolete", "superseded"]:
        if status in lowered:
            return status
    return None


def infer_policy_name(source_file: str, title_text: str, document_type: str) -> Optional[str]:
    if document_type not in {"policy", "standard", "procedure", "guidance", "methodology"}:
        return None
    for candidate in [title_text, Path(source_file).stem.replace("_", " ")]:
        cleaned = clean_title_candidate(candidate)
        if cleaned:
            return cleaned[:120]
    return None


def infer_policy_metadata(profile_text: str) -> Dict[str, str]:
    lowered = profile_text.lower()
    risk_domains = []
    validation_topics = []
    triage_topics = []
    lifecycle_stage = "unknown"

    domain_map = {
        "explainability": ["explainability", "interpretability"],
        "fairness": ["fairness", "bias"],
        "monitoring": ["monitoring", "drift", "performance monitoring"],
        "governance": ["governance", "oversight", "accountability"],
        "data_quality": ["data quality", "data governance", "data lineage"],
        "security_privacy": ["privacy", "security", "cyber"],
    }
    for name, markers in domain_map.items():
        if any(marker in lowered for marker in markers):
            risk_domains.append(name)

    topic_map = {
        "validation_scope": ["scope", "validation scope", "independent validation"],
        "risk_triage": ["triage", "risk assessment", "materiality"],
        "required_evidence": ["evidence", "documentation requirements", "required documentation"],
        "controls": ["controls", "control effectiveness"],
        "monitoring": ["monitoring", "ongoing monitoring"],
    }
    for name, markers in topic_map.items():
        if any(marker in lowered for marker in markers):
            validation_topics.append(name)
    for name, markers in {"materiality": ["materiality"], "owner_roles": ["roles and responsibilities", "owner"], "deployment": ["go live", "deployment"]}.items():
        if any(marker in lowered for marker in markers):
            triage_topics.append(name)

    for stage, markers in {
        "design": ["design", "development"],
        "pre_deployment": ["pre-deployment", "before deployment", "go live"],
        "post_deployment": ["monitoring", "ongoing"],
    }.items():
        if any(marker in lowered for marker in markers):
            lifecycle_stage = stage
            break

    return {
        "risk_domain": ", ".join(sorted(risk_domains)) if risk_domains else "unknown",
        "validation_topic": ", ".join(sorted(validation_topics)) if validation_topics else "unknown",
        "triage_topic": ", ".join(sorted(triage_topics)) if triage_topics else "unknown",
        "lifecycle_stage": lifecycle_stage,
    }


def infer_model_metadata(profile_text: str) -> Dict[str, str]:
    lowered = profile_text.lower()
    section_tags = []
    evidence_topics = []

    for tag, markers in {
        "inputs": ["input", "source data", "features"],
        "outputs": ["output", "prediction", "score"],
        "assumptions": ["assumption", "assumptions"],
        "limitations": ["limitation", "limitations", "constraint"],
        "controls": ["control", "controls"],
        "monitoring": ["monitoring", "drift", "performance"],
        "implementation": ["implementation", "architecture", "deployment"],
    }.items():
        if any(marker in lowered for marker in markers):
            section_tags.append(tag)

    for topic, markers in {
        "completeness": ["required documentation", "completeness", "missing information"],
        "methodology": ["methodology", "algorithm", "approach"],
        "governance": ["owner", "approval", "governance"],
        "monitoring": ["monitoring", "ongoing performance"],
    }.items():
        if any(marker in lowered for marker in markers):
            evidence_topics.append(topic)

    return {
        "section_type": ", ".join(sorted(section_tags)) if section_tags else "unknown",
        "evidence_topic": ", ".join(sorted(evidence_topics)) if evidence_topics else "unknown",
    }


def extract_markdown_sections(markdown_text: str) -> List[Dict[str, Any]]:
    text = markdown_text or ""
    lines = text.splitlines()
    sections: List[Dict[str, Any]] = []
    current_title = "Document Overview"
    current_level = 1
    current_lines: List[str] = []
    path_stack: List[Tuple[int, str]] = []

    def flush_section():
        body = "\n".join(current_lines).strip()
        if not body:
            return
        section_path = " > ".join(title for _, title in path_stack) if path_stack else current_title
        sections.append({
            "section_title": current_title,
            "section_path": section_path,
            "text": body[:MAX_SECTION_DOC_CHARS],
            "level": current_level,
        })

    for raw_line in lines:
        line = raw_line.strip()
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            flush_section()
            current_lines = []
            current_level = len(header_match.group(1))
            current_title = normalize_text_for_display(header_match.group(2))
            path_stack = [item for item in path_stack if item[0] < current_level]
            path_stack.append((current_level, current_title))
        else:
            current_lines.append(raw_line)

    flush_section()

    if sections:
        return sections

    normalized = normalize_text_for_display(text)
    if not normalized:
        return []
    return [{
        "section_title": "Document Overview",
        "section_path": "Document Overview",
        "text": normalized[:MAX_SECTION_DOC_CHARS],
        "level": 1,
    }]


def split_pdf_to_pages(pdf_path: str, out_dir: str) -> List[Tuple[int, str]]:
    reader = PdfReader(pdf_path)
    output: List[Tuple[int, str]] = []
    source_stem = safe_stem(pdf_path)
    for page_number, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        out_path = os.path.join(out_dir, f"{source_stem}__page_{page_number:04d}.pdf")
        with open(out_path, "wb") as handle:
            writer.write(handle)
        output.append((page_number, out_path))
    return output


def convert_page_pdf_to_markdown(converter: DocumentConverter, page_pdf_path: str, md_path: str) -> str:
    result = converter.convert(page_pdf_path)
    markdown_text = result.document.export_to_markdown()
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_text)
    return markdown_text


# =============================================================================
# Builder
# =============================================================================

class PreparedKnowledgeBaseBuilder:
    def __init__(self, profile: str, pdf_dir: str, work_root: str):
        if profile not in PROFILE_CONFIGS:
            raise ValueError(f"Unsupported profile: {profile}. Choose from {list(PROFILE_CONFIGS)}")
        self.profile = profile
        self.pdf_dir = pdf_dir
        self.work_root = work_root
        self.collection_name = PROFILE_CONFIGS[profile]["collection_name"]

        self.page_pdf_dir = os.path.join(work_root, profile, "page_pdfs")
        self.page_md_dir = os.path.join(work_root, profile, "page_markdown")
        self.chroma_dir = os.path.join(work_root, profile, "chroma_db")
        self.manifest_path = os.path.join(work_root, profile, "manifest.json")
        self.stats_path = os.path.join(work_root, profile, "kb_stats.json")

        for path in [self.page_pdf_dir, self.page_md_dir, self.chroma_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)

        self.manifest_documents: Dict[str, Any] = {}
        self.manifest_meta: Dict[str, Any] = {}
        self.documents: List[Document] = []
        self.summary_documents: List[Document] = []
        self.window_nodes = []
        self.standard_nodes = []
        self.vector_index = None
        self.ingestion_diagnostics: List[str] = []

    def reset_workspace(self):
        target_dir = os.path.join(self.work_root, self.profile)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)
        for path in [self.page_pdf_dir, self.page_md_dir, self.chroma_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def configure_models(self):
        register_cortex_provider()
        Settings.llm = LiteLLM(
            model=CORTEX_LLM_MODEL,
            temperature=CORTEX_LLM_TEMPERATURE,
            max_tokens=CORTEX_LLM_MAX_TOKENS,
            context_window=CORTEX_CONTEXT_WINDOW,
        )
        Settings.embed_model = CortexLiteLLMEmbedding(
            model_name=CORTEX_EMBED_MODEL,
            custom_llm_provider=CORTEX_CUSTOM_LLM_PROVIDER,
        )

    def _build_docling_converter(self) -> DocumentConverter:
        artifacts_path = Path(DOCLING_ARTIFACTS_PATH).expanduser().resolve()
        if DOCLING_ENFORCE_LOCAL_MODELS and not artifacts_path.exists():
            raise FileNotFoundError(
                f"Docling local artifacts path not found: {artifacts_path}. Set DOCLING_ARTIFACTS_PATH correctly."
            )

        pipeline_options = PdfPipelineOptions(artifacts_path=str(artifacts_path))
        if hasattr(pipeline_options, "do_ocr"):
            pipeline_options.do_ocr = DOCLING_ENABLE_OCR
        if hasattr(pipeline_options, "do_table_structure"):
            pipeline_options.do_table_structure = DOCLING_ENABLE_TABLE_STRUCTURE

        if DOCLING_ENABLE_OCR:
            rapidocr_paths = resolve_rapidocr_model_paths(artifacts_path)
            required = ["det_model_path", "rec_model_path", "cls_model_path"]
            if not all(k in rapidocr_paths for k in required):
                raise FileNotFoundError(
                    "Docling OCR is enabled but RapidOCR local models were not fully found. "
                    "Set RAPIDOCR_MODELS_PATH or the explicit RAPIDOCR_*_MODEL_PATH variables."
                )
            pipeline_options.ocr_options = RapidOcrOptions(
                det_model_path=str(rapidocr_paths["det_model_path"]),
                rec_model_path=str(rapidocr_paths["rec_model_path"]),
                cls_model_path=str(rapidocr_paths["cls_model_path"]),
            )

        log(
            f"Docling pipeline config: ocr={'enabled' if DOCLING_ENABLE_OCR else 'disabled'}, "
            f"table_structure={'enabled' if DOCLING_ENABLE_TABLE_STRUCTURE else 'disabled'}"
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def _extract_doc_profile(self, source_file: str, source_path: str, text: str) -> Dict[str, Any]:
        lines = (text or "").splitlines()
        cleaned_lines = []
        for line in lines[:40]:
            candidate = clean_title_candidate(line)
            if candidate:
                cleaned_lines.append(candidate)

        title_line = cleaned_lines[0] if cleaned_lines else clean_title_candidate(Path(source_file).stem.replace("_", " "))
        early_text = "\n".join(cleaned_lines[:10]) if cleaned_lines else (text or "")[:800]
        filename_text = f"{source_file} {Path(source_path).stem}".replace("_", " ").replace("-", " ")
        profile_text = f"{filename_text}\n{title_line}\n{early_text}\n{text[:1200]}"

        document_type = infer_document_type(profile_text, self.profile)
        profile = {
            "document_type": document_type,
            "policy_name": infer_policy_name(source_file, title_line, document_type),
            "model_name": infer_model_name(profile_text) if self.profile == "model_evidence" else "unknown",
            "version": parse_version(profile_text),
            "effective_date": parse_effective_date(profile_text),
            "owner": infer_owner(profile_text),
            "business_area": infer_business_area(source_file, title_line, early_text, self.profile),
            "approval_status": infer_approval_status(profile_text),
            "title_text": title_line or Path(source_file).stem,
        }

        if self.profile == "policy_guidance":
            profile.update(infer_policy_metadata(profile_text))
            profile.update({"section_type": "unknown", "evidence_topic": "unknown"})
        else:
            profile.update(infer_model_metadata(profile_text))
            profile.update({"risk_domain": "unknown", "validation_topic": "unknown", "triage_topic": "unknown", "lifecycle_stage": "unknown"})

        return {key: (value if value not in {"", None} else "unknown") for key, value in profile.items()}

    def _build_base_metadata(self, source_file: str, source_path: str, doc_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "kb_profile": self.profile,
            "source_file": source_file,
            "source_path": source_path,
            "doc_id": doc_id,
            "file_type": "pdf",
            "document_type": profile.get("document_type", "unknown"),
            "policy_name": profile.get("policy_name", "unknown"),
            "model_name": profile.get("model_name", "unknown"),
            "version": profile.get("version", "unknown"),
            "effective_date": profile.get("effective_date", "unknown"),
            "owner": profile.get("owner", "unknown"),
            "business_area": profile.get("business_area", "unknown"),
            "approval_status": profile.get("approval_status", "unknown"),
            "risk_domain": profile.get("risk_domain", "unknown"),
            "validation_topic": profile.get("validation_topic", "unknown"),
            "triage_topic": profile.get("triage_topic", "unknown"),
            "lifecycle_stage": profile.get("lifecycle_stage", "unknown"),
            "section_type": profile.get("section_type", "unknown"),
            "evidence_topic": profile.get("evidence_topic", "unknown"),
            "section_title": "Document Overview",
            "section_path": "Document Overview",
            "title_text": profile.get("title_text", Path(source_file).stem),
        }

    def _add_document_diagnostic(self, source_file: str, message: str):
        self.ingestion_diagnostics.append(f"{source_file}: {message}")

    def _build_section_documents(self, markdown_text: str, base_metadata: Dict[str, Any], page_number: Optional[int] = None) -> List[Document]:
        section_docs: List[Document] = []
        for index, section in enumerate(extract_markdown_sections(markdown_text), start=1):
            section_text = (section.get("text") or "").strip()
            if len(normalize_for_hash(section_text)) < 20:
                continue
            metadata = dict(base_metadata)
            metadata.update({
                "page_number": page_number,
                "section_title": section.get("section_title", "Document Overview"),
                "section_path": section.get("section_path", "Document Overview"),
                "section_id": f"{base_metadata.get('doc_id')}__sec_{index:03d}" + (f"_p{page_number:04d}" if page_number else ""),
                "citation_label": f"{base_metadata.get('source_file')} p.{page_number} - {section.get('section_title', 'Section')}",
                "retrieval_style": "section_summary",
            })
            # lightweight section typing
            section_text_lower = section_text.lower()
            if self.profile == "policy_guidance":
                tags = []
                if any(x in section_text_lower for x in ["scope", "validation scope"]): tags.append("scope")
                if any(x in section_text_lower for x in ["triage", "materiality", "risk assessment"]): tags.append("triage")
                if any(x in section_text_lower for x in ["control", "controls"]): tags.append("controls")
                if any(x in section_text_lower for x in ["governance", "oversight", "responsibility"]): tags.append("governance")
                metadata["section_type"] = ", ".join(tags) if tags else metadata.get("section_type", "unknown")
            else:
                tags = []
                if any(x in section_text_lower for x in ["input", "feature", "source data"]): tags.append("inputs")
                if any(x in section_text_lower for x in ["output", "prediction", "score"]): tags.append("outputs")
                if any(x in section_text_lower for x in ["assumption", "assumptions"]): tags.append("assumptions")
                if any(x in section_text_lower for x in ["limitation", "limitations"]): tags.append("limitations")
                if any(x in section_text_lower for x in ["monitoring", "performance"]): tags.append("monitoring")
                if any(x in section_text_lower for x in ["control", "controls"]): tags.append("controls")
                metadata["section_type"] = ", ".join(tags) if tags else metadata.get("section_type", "unknown")
            section_docs.append(Document(text=section_text, metadata=metadata))
        return section_docs

    def ingest_documents(self):
        pdf_dir = Path(self.pdf_dir)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF input folder not found: {pdf_dir.resolve()}")

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        log("program start")
        log(f"KB profile: {self.profile}")
        log(f"number of PDF files found: {len(pdf_files)}")
        if not pdf_files:
            raise RuntimeError(f"No PDF files found in {pdf_dir.resolve()}")

        self.manifest_documents = {}
        self.documents = []
        self.summary_documents = []
        self.ingestion_diagnostics = []

        converter = self._build_docling_converter()
        log(f"Docling local artifacts path: {Path(DOCLING_ARTIFACTS_PATH).expanduser()}")

        seen_page_hashes: Dict[str, str] = {}
        seen_doc_hashes: Dict[str, str] = {}

        for file_index, pdf_path in enumerate(pdf_files, start=1):
            source_file = pdf_path.name
            source_path = str(pdf_path.resolve())
            source_hash = file_md5(source_path)
            file_diagnostics: List[str] = []

            log(f"current file being processed [{file_index}/{len(pdf_files)}]: {source_file}")
            page_pdfs = split_pdf_to_pages(source_path, self.page_pdf_dir)
            log(f"PDF page split progress: created {len(page_pdfs)} page PDFs for {source_file}")

            page_records = []
            first_page_text = ""
            page_level_doc_count = 0
            page_summary_doc_count = 0

            for page_number, page_pdf_path in page_pdfs:
                doc_id = f"{safe_stem(source_path)}__p{page_number:04d}"
                page_md_path = os.path.join(self.page_md_dir, f"{doc_id}.md")

                log(f"Docling conversion progress: {source_file} page {page_number}/{len(page_pdfs)}")
                text = convert_page_pdf_to_markdown(converter, page_pdf_path, page_md_path).strip()

                if page_number == 1:
                    first_page_text = text

                normalized_hash = hashlib.md5(normalize_for_hash(text).encode("utf-8")).hexdigest() if text else None
                if normalized_hash and normalized_hash in seen_page_hashes:
                    file_diagnostics.append(f"page {page_number} duplicates content from {seen_page_hashes[normalized_hash]}")
                elif normalized_hash:
                    seen_page_hashes[normalized_hash] = f"{source_file} p.{page_number}"

                if len(normalize_text_for_display(text)) < MIN_MEANINGFUL_TEXT_LEN:
                    file_diagnostics.append(f"page {page_number} is empty or nearly empty")

                profile = self._extract_doc_profile(source_file, source_path, first_page_text or text)
                base_metadata = self._build_base_metadata(source_file, source_path, doc_id, profile)
                metadata = dict(base_metadata)
                metadata.update({
                    "page_number": page_number,
                    "page_markdown_path": page_md_path,
                    "citation_label": f"{source_file} p.{page_number}",
                })

                if text:
                    sections = extract_markdown_sections(text)
                    metadata["section_title"] = sections[0]["section_title"] if sections else "Document Overview"
                    metadata["section_path"] = sections[0]["section_path"] if sections else "Document Overview"
                    self.documents.append(Document(text=text, metadata=metadata))
                    page_level_doc_count += 1

                    section_docs = self._build_section_documents(text, metadata, page_number=page_number)
                    self.summary_documents.extend(section_docs)
                    page_summary_doc_count += len(section_docs)

                page_records.append({
                    "page_number": page_number,
                    "page_pdf_path": page_pdf_path,
                    "page_markdown_path": page_md_path,
                    "doc_id": doc_id,
                    "metadata": metadata,
                })

            source_profile = self._extract_doc_profile(source_file, source_path, first_page_text)
            file_base_metadata = self._build_base_metadata(
                source_file=source_file,
                source_path=source_path,
                doc_id=safe_stem(source_path),
                profile=source_profile,
            )

            if source_hash in seen_doc_hashes:
                file_diagnostics.append(f"duplicate file content detected with {seen_doc_hashes[source_hash]}")
            else:
                seen_doc_hashes[source_hash] = source_file

            self.manifest_documents[source_path] = {
                "source_file": source_file,
                "file_type": "pdf",
                "md5": source_hash,
                "pages": page_records,
                "base_metadata": file_base_metadata,
                "diagnostics": sorted(set(file_diagnostics)),
            }

            for warning in sorted(set(file_diagnostics)):
                self._add_document_diagnostic(source_file, warning)

            log(f"number of page-level documents produced for {source_file}: {page_level_doc_count}")
            log(f"number of summary/section documents produced for {source_file}: {page_summary_doc_count}")

        self.manifest_meta = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "kb_profile": self.profile,
            "embedding_model": CORTEX_EMBED_MODEL,
            "embedding_provider": CORTEX_CUSTOM_LLM_PROVIDER,
            "llm_model": CORTEX_LLM_MODEL,
            "context_window": CORTEX_CONTEXT_WINDOW,
            "pdf_dir": str(Path(self.pdf_dir).resolve()),
        }

        save_json(self.manifest_path, {"meta": self.manifest_meta, "documents": self.manifest_documents})

        if DELETE_TEMP_PAGE_PDFS:
            shutil.rmtree(self.page_pdf_dir, ignore_errors=True)
            Path(self.page_pdf_dir).mkdir(parents=True, exist_ok=True)

    def build_nodes(self):
        log("building sentence-window nodes")
        node_parser = SentenceWindowNodeParser.from_defaults(
            sentence_splitter=english_sentence_splitter,
            window_size=WINDOW_SIZE,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        self.window_nodes = node_parser.get_nodes_from_documents(self.documents)
        for node in self.window_nodes:
            if node.metadata is None:
                node.metadata = {}
            node.metadata["retrieval_style"] = "sentence_window"
            node.excluded_embed_metadata_keys = []
            node.excluded_llm_metadata_keys = []
        log(f"number of sentence-window nodes produced: {len(self.window_nodes)}")

        log("building standard chunk nodes")
        splitter = SentenceSplitter(chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_CHUNK_OVERLAP)
        self.standard_nodes = splitter.get_nodes_from_documents(self.documents)
        for node in self.standard_nodes:
            if node.metadata is None:
                node.metadata = {}
            node.metadata["retrieval_style"] = "standard_chunk"
            node.excluded_embed_metadata_keys = []
            node.excluded_llm_metadata_keys = []
        log(f"number of standard chunks produced: {len(self.standard_nodes)}")

    def build_index(self):
        log("vector database creation started")
        self.configure_models()
        shutil.rmtree(self.chroma_dir, ignore_errors=True)
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)

        chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.vector_index = VectorStoreIndex(self.window_nodes, storage_context=storage_context)
        log("vector database insertion completed")
        log(f"total documents indexed: {len(self.documents)}")
        log(f"total summary documents indexed: {len(self.summary_documents)}")
        log(f"total nodes indexed in vector database: {chroma_collection.count()}")

    def save_stats(self):
        metadata_fields = sorted({key for doc in self.documents for key in (doc.metadata or {}).keys()})
        stats = {
            "kb_profile": self.profile,
            "pdfdata_dir": str(Path(self.pdf_dir).resolve()),
            "work_dir": str(Path(self.work_root).resolve()),
            "chroma_dir": str(Path(self.chroma_dir).resolve()),
            "collection_name": self.collection_name,
            "source_pdf_count": len(self.manifest_documents),
            "page_level_document_count": len(self.documents),
            "summary_document_count": len(self.summary_documents),
            "standard_chunk_count": len(self.standard_nodes),
            "sentence_window_node_count": len(self.window_nodes),
            "metadata_fields": metadata_fields,
            "ingestion_diagnostics": self.ingestion_diagnostics,
            "embedding_model": CORTEX_EMBED_MODEL,
            "embedding_provider": CORTEX_CUSTOM_LLM_PROVIDER,
            "llm_model": CORTEX_LLM_MODEL,
            "context_window": CORTEX_CONTEXT_WINDOW,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        }
        save_json(self.stats_path, stats)
        log(f"summary of metadata fields preserved: {', '.join(metadata_fields)}")

    def run_verification_tests(self):
        log("verification started")
        chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        chroma_collection = chroma_client.get_collection(self.collection_name)
        queries = VERIFICATION_QUERIES[self.profile]

        print()
        print("=== Verification Summary ===")
        print(f"KB profile: {self.profile}")
        print(f"Vector database path: {Path(self.chroma_dir).resolve()}")
        print(f"Collection name: {self.collection_name}")
        print(f"Collection exists: {Path(self.chroma_dir).exists()}")
        print(f"Total indexed vector nodes/documents: {chroma_collection.count()}")
        print(f"Total page-level documents: {len(self.documents)}")
        print(f"Total standard chunks: {len(self.standard_nodes)}")
        print(f"Total sentence-window nodes: {len(self.window_nodes)}")
        print(f"Manifest saved: {Path(self.manifest_path).exists()}")
        print(f"Stats saved: {Path(self.stats_path).exists()}")
        print(f"Embedding model: {CORTEX_EMBED_MODEL}")
        print(f"Embedding provider: {CORTEX_CUSTOM_LLM_PROVIDER}")

        print()
        print("Example metadata records:")
        for index, doc in enumerate(self.documents[:3], start=1):
            sample_meta = dict(sorted((doc.metadata or {}).items()))
            print(f"{index}. {json.dumps(sample_meta, ensure_ascii=False)}")

        print()
        print("Metadata presence checks:")
        for key in ["source_file", "page_number", "citation_label", "document_type", "kb_profile"]:
            present = any(doc.metadata.get(key) for doc in self.documents if doc.metadata) if key != "page_number" else any(doc.metadata.get(key) is not None for doc in self.documents if doc.metadata)
            print(f"- {key}: {present}")

        print()
        print(f"Simple vector similarity test query: {queries['vector']}")
        vector_retriever = VectorIndexRetriever(index=self.vector_index, similarity_top_k=VECTOR_TOP_K)
        vector_results = vector_retriever.retrieve(queries['vector'])
        print(f"Vector retrieved nodes: {len(vector_results)}")
        for index, node_with_score in enumerate(vector_results[:3], start=1):
            metadata = node_with_score.node.metadata or {}
            print(f"- [{index}] {metadata.get('citation_label')} | score={getattr(node_with_score, 'score', None)} | text={short_preview(node_with_score.node.text)}")

        print()
        print(f"Simple BM25 test query: {queries['bm25']}")
        bm25_retriever = BM25Retriever.from_defaults(nodes=self.standard_nodes, similarity_top_k=BM25_TOP_K)
        bm25_results = bm25_retriever.retrieve(queries['bm25'])
        print(f"BM25 retrieved nodes: {len(bm25_results)}")
        for index, node_with_score in enumerate(bm25_results[:3], start=1):
            metadata = node_with_score.node.metadata or {}
            print(f"- [{index}] {metadata.get('citation_label')} | score={getattr(node_with_score, 'score', None)} | text={short_preview(node_with_score.node.text)}")

        print()
        print("Sample retrieved chunks with source/page metadata:")
        for node_with_score in vector_results[:3]:
            metadata = node_with_score.node.metadata or {}
            print({
                "source_file": metadata.get("source_file"),
                "page_number": metadata.get("page_number"),
                "doc_id": metadata.get("doc_id"),
                "retrieval_style": metadata.get("retrieval_style"),
                "profile": metadata.get("kb_profile"),
                "preview": short_preview(node_with_score.node.text, 160),
            })

        kb_ready = (
            len(self.documents) > 0 and len(self.window_nodes) > 0 and chroma_collection.count() > 0 and Path(self.manifest_path).exists()
        )
        print()
        print(f"KB ready for querying: {kb_ready}")
        print("=== End Verification ===")
        print()

    def print_examples(self):
        print("=== Example Commands ===")
        print(f"python prepare_kb_template.py --profile {self.profile} --pdf-dir {self.pdf_dir}")
        print(f"python chat_kb_template.py --profile {self.profile} --work-root {self.work_root}")
        print("========================")

    def prepare(self):
        self.reset_workspace()
        self.ingest_documents()
        if not self.documents:
            raise RuntimeError("No page-level documents were produced from the PDFs.")
        self.build_nodes()
        self.build_index()
        self.save_stats()
        log("final completion message: knowledge base preparation completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Prepare a profile-based RAG knowledge base.")
    parser.add_argument("--profile", choices=list(PROFILE_CONFIGS.keys()), required=True)
    parser.add_argument("--pdf-dir", default=None, help="Folder containing PDFs for this KB profile.")
    parser.add_argument("--work-root", default="./kb_workspaces", help="Root folder for KB artifacts.")
    args = parser.parse_args()

    pdf_dir = args.pdf_dir or PROFILE_CONFIGS[args.profile]["default_pdf_dir"]
    builder = PreparedKnowledgeBaseBuilder(profile=args.profile, pdf_dir=pdf_dir, work_root=args.work_root)
    builder.prepare()
    builder.run_verification_tests()
    builder.print_examples()


if __name__ == "__main__":
    main()
