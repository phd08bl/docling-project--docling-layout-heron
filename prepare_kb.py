import os
import re
import json
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import chromadb
from docling.document_converter import DocumentConverter
from pypdf import PdfReader, PdfWriter

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore


# =============================================================================
# Config
# =============================================================================

PDFDATA_DIR = "./PDFdata"
WORK_DIR = "./kb_terminal_workspace"
PAGE_PDF_DIR = os.path.join(WORK_DIR, "page_pdfs")
PAGE_MD_DIR = os.path.join(WORK_DIR, "page_markdown")
CHROMA_DIR = os.path.join(WORK_DIR, "chroma_db")
MANIFEST_PATH = os.path.join(WORK_DIR, "manifest.json")
KB_STATS_PATH = os.path.join(WORK_DIR, "kb_stats.json")
COLLECTION_NAME = "kb_terminal_collection"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "qwen3:8b"
OLLAMA_EMBED_MODEL = "embeddinggemma"

BASE_CHUNK_SIZE = 900
BASE_CHUNK_OVERLAP = 120
WINDOW_SIZE = 3

VECTOR_TOP_K = 4
BM25_TOP_K = 4
MIN_MEANINGFUL_TEXT_LEN = 80
MAX_SECTION_DOC_CHARS = 3200
DELETE_TEMP_PAGE_PDFS = False

VERIFICATION_VECTOR_QUERY = "Summarize the main purpose of this document."
VERIFICATION_BM25_QUERY = "policy requirements responsibilities controls"


# =============================================================================
# Helpers
# =============================================================================

def log(message: str):
    print(f"[prepare_kb] {message}", flush=True)


def ensure_dirs():
    for path in [WORK_DIR, PAGE_PDF_DIR, PAGE_MD_DIR, CHROMA_DIR]:
        Path(path).mkdir(parents=True, exist_ok=True)


def reset_workspace():
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    ensure_dirs()


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


def safe_meta_value(value: Optional[str]) -> str:
    value = normalize_text_for_display(str(value or ""))
    return value if value else "unknown"


def parse_version(text: str) -> Optional[str]:
    match = re.search(r"\b(?:version|ver|v)[\s._-]*([0-9]+(?:\.[0-9]+){0,2})\b", text, flags=re.I)
    return match.group(1) if match else None


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


def infer_document_type(text: str) -> str:
    lowered = (text or "").lower()
    mapping = [
        ("validation_report", ["validation report", "validator report", "independent review", "validation"]),
        ("model_document", ["model development", "model document", "model inventory", "model overview", "model specification"]),
        ("methodology", ["methodology", "approach", "method", "framework"]),
        ("procedure", ["procedure", "workflow", "process"]),
        ("policy", ["policy"]),
        ("standard", ["standard"]),
        ("appendix", ["appendix", "annex"]),
    ]
    for doc_type, markers in mapping:
        if any(marker in lowered for marker in markers):
            return doc_type
    return "unknown"


def infer_business_area(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    areas = {
        "credit_risk": ["credit", "loan", "pd", "lgd", "ead"],
        "market_risk": ["market risk", "var", "stressed var"],
        "fraud": ["fraud"],
        "compliance": ["compliance", "aml", "kyc"],
        "finance": ["finance", "ifrs", "cecl"],
        "retail_banking": ["retail", "consumer"],
    }
    for area, markers in areas.items():
        if any(marker in lowered for marker in markers):
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
    if document_type not in {"policy", "standard", "procedure", "methodology"}:
        return None
    for candidate in [title_text, Path(source_file).stem]:
        cleaned = normalize_text_for_display(candidate)
        if cleaned:
            return cleaned[:120]
    return None


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


class PreparedKnowledgeBaseBuilder:
    def __init__(self):
        ensure_dirs()
        self.manifest: Dict[str, Any] = {}
        self.documents: List[Document] = []
        self.summary_documents: List[Document] = []
        self.window_nodes = []
        self.standard_nodes = []
        self.vector_index = None
        self.ingestion_diagnostics: List[str] = []

    def configure_models(self):
        Settings.llm = Ollama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=300.0,
            temperature=0.1,
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

    def _extract_doc_profile(self, source_file: str, source_path: str, text: str) -> Dict[str, Any]:
        title_line = ""
        for line in (text or "").splitlines():
            clean = normalize_text_for_display(line)
            if clean:
                title_line = clean
                break

        filename_text = f"{source_file} {Path(source_path).stem}".replace("_", " ").replace("-", " ")
        profile_text = f"{filename_text}\n{title_line}\n{text[:1500]}"
        document_type = infer_document_type(profile_text)

        profile = {
            "document_type": document_type,
            "policy_name": infer_policy_name(source_file, title_line, document_type),
            "model_name": infer_model_name(profile_text),
            "version": parse_version(profile_text),
            "effective_date": parse_effective_date(profile_text),
            "owner": infer_owner(profile_text),
            "business_area": infer_business_area(profile_text),
            "approval_status": infer_approval_status(profile_text),
            "title_text": title_line or Path(source_file).stem,
        }
        return {key: (value if value not in {"", None} else "unknown") for key, value in profile.items()}

    def _build_base_metadata(
        self,
        source_file: str,
        source_path: str,
        doc_id: str,
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
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
            "section_title": "Document Overview",
            "section_path": "Document Overview",
            "title_text": profile.get("title_text", Path(source_file).stem),
        }

    def _add_document_diagnostic(self, source_file: str, message: str):
        self.ingestion_diagnostics.append(f"{source_file}: {message}")

    def _build_section_documents(
        self,
        markdown_text: str,
        base_metadata: Dict[str, Any],
        page_number: Optional[int] = None,
    ) -> List[Document]:
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
            section_docs.append(Document(text=section_text, metadata=metadata))
        return section_docs

    def ingest_documents(self):
        pdf_dir = Path(PDFDATA_DIR)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF input folder not found: {pdf_dir.resolve()}")

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        log("program start")
        log(f"number of PDF files found: {len(pdf_files)}")

        if not pdf_files:
            raise RuntimeError(f"No PDF files found in {pdf_dir.resolve()}")

        self.manifest = {}
        self.documents = []
        self.summary_documents = []
        self.ingestion_diagnostics = []

        converter = DocumentConverter()
        seen_page_hashes: Dict[str, str] = {}
        seen_doc_hashes: Dict[str, str] = {}

        for file_index, pdf_path in enumerate(pdf_files, start=1):
            source_file = pdf_path.name
            source_path = str(pdf_path.resolve())
            source_hash = file_md5(source_path)
            file_diagnostics: List[str] = []

            log(f"current file being processed [{file_index}/{len(pdf_files)}]: {source_file}")
            page_pdfs = split_pdf_to_pages(source_path, PAGE_PDF_DIR)
            log(f"PDF page split progress: created {len(page_pdfs)} page PDFs for {source_file}")

            page_records = []
            first_page_text = ""
            page_level_doc_count = 0
            page_summary_doc_count = 0

            for page_number, page_pdf_path in page_pdfs:
                doc_id = f"{safe_stem(source_path)}__p{page_number:04d}"
                page_md_path = os.path.join(PAGE_MD_DIR, f"{doc_id}.md")

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

            self.manifest[source_path] = {
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

        save_json(MANIFEST_PATH, self.manifest)
        if DELETE_TEMP_PAGE_PDFS:
            shutil.rmtree(PAGE_PDF_DIR, ignore_errors=True)
            Path(PAGE_PDF_DIR).mkdir(parents=True, exist_ok=True)

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
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.vector_index = VectorStoreIndex(self.window_nodes, storage_context=storage_context)
        log("vector database insertion completed")
        log(f"total documents indexed: {len(self.documents)}")
        log(f"total summary documents indexed: {len(self.summary_documents)}")
        log(f"total nodes indexed in vector database: {chroma_collection.count()}")

    def save_stats(self):
        metadata_fields = sorted({
            key
            for doc in self.documents
            for key in (doc.metadata or {}).keys()
        })
        stats = {
            "pdfdata_dir": str(Path(PDFDATA_DIR).resolve()),
            "work_dir": str(Path(WORK_DIR).resolve()),
            "chroma_dir": str(Path(CHROMA_DIR).resolve()),
            "collection_name": COLLECTION_NAME,
            "source_pdf_count": len(self.manifest),
            "page_level_document_count": len(self.documents),
            "summary_document_count": len(self.summary_documents),
            "standard_chunk_count": len(self.standard_nodes),
            "sentence_window_node_count": len(self.window_nodes),
            "metadata_fields": metadata_fields,
            "ingestion_diagnostics": self.ingestion_diagnostics,
        }
        save_json(KB_STATS_PATH, stats)
        log(f"summary of metadata fields preserved: {', '.join(metadata_fields)}")

    def run_verification_tests(self):
        log("verification started")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        chroma_collection = chroma_client.get_collection(COLLECTION_NAME)

        print()
        print("=== Verification Summary ===")
        print(f"Vector database path: {Path(CHROMA_DIR).resolve()}")
        print(f"Collection name: {COLLECTION_NAME}")
        print(f"Collection exists: {Path(CHROMA_DIR).exists()}")
        print(f"Total indexed vector nodes/documents: {chroma_collection.count()}")
        print(f"Total page-level documents: {len(self.documents)}")
        print(f"Total standard chunks: {len(self.standard_nodes)}")
        print(f"Total sentence-window nodes: {len(self.window_nodes)}")
        print(f"Manifest saved: {Path(MANIFEST_PATH).exists()}")
        print(f"Stats saved: {Path(KB_STATS_PATH).exists()}")

        print()
        print("Example metadata records:")
        for index, doc in enumerate(self.documents[:3], start=1):
            sample_meta = dict(sorted((doc.metadata or {}).items()))
            print(f"{index}. {json.dumps(sample_meta, ensure_ascii=False)}")

        print()
        print("Metadata presence checks:")
        metadata_checks = {
            "source_file": any(doc.metadata.get("source_file") for doc in self.documents if doc.metadata),
            "page_number": any(doc.metadata.get("page_number") is not None for doc in self.documents if doc.metadata),
            "citation_label": any(doc.metadata.get("citation_label") for doc in self.documents if doc.metadata),
            "document_type": any(doc.metadata.get("document_type") for doc in self.documents if doc.metadata),
        }
        for key, present in metadata_checks.items():
            print(f"- {key}: {present}")

        print()
        print(f"Simple vector similarity test query: {VERIFICATION_VECTOR_QUERY}")
        vector_retriever = VectorIndexRetriever(index=self.vector_index, similarity_top_k=VECTOR_TOP_K)
        vector_results = vector_retriever.retrieve(VERIFICATION_VECTOR_QUERY)
        print(f"Vector retrieved nodes: {len(vector_results)}")
        for index, node_with_score in enumerate(vector_results[:3], start=1):
            metadata = node_with_score.node.metadata or {}
            print(
                f"- [{index}] {metadata.get('citation_label')} | "
                f"score={getattr(node_with_score, 'score', None)} | "
                f"text={short_preview(node_with_score.node.text)}"
            )

        print()
        print(f"Simple BM25 test query: {VERIFICATION_BM25_QUERY}")
        bm25_retriever = BM25Retriever.from_defaults(nodes=self.standard_nodes, similarity_top_k=BM25_TOP_K)
        bm25_results = bm25_retriever.retrieve(VERIFICATION_BM25_QUERY)
        print(f"BM25 retrieved nodes: {len(bm25_results)}")
        for index, node_with_score in enumerate(bm25_results[:3], start=1):
            metadata = node_with_score.node.metadata or {}
            print(
                f"- [{index}] {metadata.get('citation_label')} | "
                f"score={getattr(node_with_score, 'score', None)} | "
                f"text={short_preview(node_with_score.node.text)}"
            )

        print()
        print("Sample retrieved chunks with source/page metadata:")
        for node_with_score in vector_results[:3]:
            metadata = node_with_score.node.metadata or {}
            print({
                "source_file": metadata.get("source_file"),
                "page_number": metadata.get("page_number"),
                "doc_id": metadata.get("doc_id"),
                "retrieval_style": metadata.get("retrieval_style"),
                "preview": short_preview(node_with_score.node.text, 160),
            })

        kb_ready = (
            len(self.documents) > 0
            and len(self.window_nodes) > 0
            and chroma_collection.count() > 0
            and Path(MANIFEST_PATH).exists()
        )
        print()
        print(f"KB ready for querying: {kb_ready}")
        print("=== End Verification ===")
        print()

    def prepare(self):
        self.ingest_documents()
        if not self.documents:
            raise RuntimeError("No page-level documents were produced from the PDFs.")
        self.build_nodes()
        self.build_index()
        self.save_stats()
        log("final completion message: knowledge base preparation completed successfully")


def main():
    reset_workspace()
    builder = PreparedKnowledgeBaseBuilder()
    builder.prepare()
    builder.run_verification_tests()


if __name__ == "__main__":
    main()
