# pip install -U docling pypdf chromadb rank_bm25 llama-index-core llama-index-llms-ollama llama-index-embeddings-ollama llama-index-retrievers-bm25 llama-index-vector-stores-chroma

import os
import re
import json
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from pypdf import PdfReader, PdfWriter

from docling.document_converter import DocumentConverter

import chromadb

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    SummaryIndex,
)
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SimilarityPostprocessor,
    LLMRerank,
)
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.core.tools import QueryEngineTool
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# =============================================================================
# Config
# =============================================================================

PDF_FOLDER = "./data"
WORK_DIR = "./kb_workspace_en"
PAGE_PDF_DIR = os.path.join(WORK_DIR, "page_pdfs")
PAGE_MD_DIR = os.path.join(WORK_DIR, "page_markdown")
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs")
CHROMA_DIR = os.path.join(WORK_DIR, "chroma_db")
MANIFEST_PATH = os.path.join(WORK_DIR, "manifest.json")

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "qwen3:8b"
OLLAMA_EMBED_MODEL = "embeddinggemma"

BASE_CHUNK_SIZE = 900
BASE_CHUNK_OVERLAP = 120

WINDOW_SIZE = 3

VECTOR_TOP_K = 8
BM25_TOP_K = 8
FUSION_TOP_K = 10
SIMILARITY_CUTOFF = 0.15

USE_RERANKER = True
RERANK_TOP_N = 5

DETAIL_RESPONSE_MODE = "compact"
SUMMARY_RESPONSE_MODE = "tree_summarize"

DELETE_TEMP_PAGE_PDFS = False


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs():
    for p in [WORK_DIR, PAGE_PDF_DIR, PAGE_MD_DIR, OUTPUT_DIR, CHROMA_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_manifest() -> Dict[str, Any]:
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(data: Dict[str, Any]):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_stem(path: str) -> str:
    return Path(path).stem.replace(" ", "_")


def english_sentence_splitter(text: str) -> List[str]:
    # Simple English-oriented sentence splitter
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]


def english_tokenizer(text: str) -> List[str]:
    # Lowercase word tokenizer for BM25
    return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())


def write_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def short_preview(text: str, n: int = 280) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:n] + ("..." if len(text) > n else "")


# =============================================================================
# PDF split
# =============================================================================

def split_pdf_to_pages(pdf_path: str, out_dir: str) -> List[Tuple[int, str]]:
    reader = PdfReader(pdf_path)
    results = []

    source_stem = safe_stem(pdf_path)

    for i, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)

        out_path = os.path.join(out_dir, f"{source_stem}__page_{i:04d}.pdf")
        with open(out_path, "wb") as f:
            writer.write(f)

        results.append((i, out_path))

    return results


# =============================================================================
# Docling conversion
# =============================================================================

def convert_page_pdf_to_markdown(converter: DocumentConverter, page_pdf_path: str, md_path: str):
    result = converter.convert(page_pdf_path)
    md_text = result.document.export_to_markdown()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    return md_text


def convert_folder_pdfs_to_page_markdown(pdf_folder: str) -> List[Dict[str, Any]]:
    ensure_dirs()
    manifest = load_manifest()
    converter = DocumentConverter()

    pdf_paths = sorted(Path(pdf_folder).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_folder}")

    records: List[Dict[str, Any]] = []

    for pdf_path in pdf_paths:
        pdf_path_str = str(pdf_path)
        pdf_hash = file_md5(pdf_path_str)
        source_file = pdf_path.name
        source_stem = safe_stem(pdf_path_str)

        reuse = False
        if pdf_path_str in manifest:
            old = manifest[pdf_path_str]
            if old.get("md5") == pdf_hash:
                existing_pages = old.get("pages", [])
                if existing_pages and all(os.path.exists(x["page_markdown_path"]) for x in existing_pages):
                    reuse = True
                    print(f"[Reuse] {source_file}")
                    for x in existing_pages:
                        records.append(x)

        if reuse:
            continue

        print(f"[Split] {source_file}")
        page_pdfs = split_pdf_to_pages(pdf_path_str, PAGE_PDF_DIR)

        page_records = []
        for page_number, page_pdf_path in page_pdfs:
            doc_id = f"{source_stem}__p{page_number:04d}"
            page_md_path = os.path.join(PAGE_MD_DIR, f"{doc_id}.md")

            print(f"  [Docling] {source_file} page {page_number}")
            convert_page_pdf_to_markdown(converter, page_pdf_path, page_md_path)

            rec = {
                "source_pdf": pdf_path_str,
                "source_file": source_file,
                "page_number": page_number,
                "page_pdf_path": page_pdf_path,
                "page_markdown_path": page_md_path,
                "doc_id": doc_id,
            }
            page_records.append(rec)
            records.append(rec)

        manifest[pdf_path_str] = {
            "md5": pdf_hash,
            "pages": page_records,
        }
        save_manifest(manifest)

    if DELETE_TEMP_PAGE_PDFS:
        shutil.rmtree(PAGE_PDF_DIR, ignore_errors=True)
        Path(PAGE_PDF_DIR).mkdir(parents=True, exist_ok=True)

    return records


# =============================================================================
# Build documents
# =============================================================================

def page_markdown_records_to_documents(records: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []

    for rec in records:
        with open(rec["page_markdown_path"], "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        metadata = {
            "source_file": rec["source_file"],
            "source_pdf": rec["source_pdf"],
            "page_number": rec["page_number"],
            "doc_id": rec["doc_id"],
            "page_markdown_path": rec["page_markdown_path"],
            "citation_label": f"{rec['source_file']} p.{rec['page_number']}",
        }

        docs.append(Document(text=text, metadata=metadata))

    return docs


# =============================================================================
# Nodes
# =============================================================================

def build_sentence_window_nodes(documents: List[Document]):
    node_parser = SentenceWindowNodeParser.from_defaults(
        sentence_splitter=english_sentence_splitter,
        window_size=WINDOW_SIZE,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    nodes = node_parser.get_nodes_from_documents(documents)

    for n in nodes:
        if n.metadata is None:
            n.metadata = {}
        n.metadata["retrieval_style"] = "sentence_window"
        n.excluded_embed_metadata_keys = []
        n.excluded_llm_metadata_keys = []

    return nodes


def build_standard_nodes(documents: List[Document]):
    splitter = SentenceSplitter(
        chunk_size=BASE_CHUNK_SIZE,
        chunk_overlap=BASE_CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    for n in nodes:
        if n.metadata is None:
            n.metadata = {}
        n.metadata["retrieval_style"] = "standard_chunk"
        n.excluded_embed_metadata_keys = []
        n.excluded_llm_metadata_keys = []

    return nodes


# =============================================================================
# Filters
# =============================================================================

def build_metadata_filters(
    source_files: Optional[List[str]] = None,
    doc_ids: Optional[List[str]] = None,
    page_numbers: Optional[List[int]] = None,
) -> Optional[MetadataFilters]:
    filters = []

    if source_files:
        for sf in source_files:
            filters.append(MetadataFilter(key="source_file", value=sf, operator=FilterOperator.EQ))

    if doc_ids:
        for did in doc_ids:
            filters.append(MetadataFilter(key="doc_id", value=did, operator=FilterOperator.EQ))

    if page_numbers:
        for p in page_numbers:
            filters.append(MetadataFilter(key="page_number", value=p, operator=FilterOperator.EQ))

    if not filters:
        return None

    return MetadataFilters(filters=filters)


def filter_documents(
    documents: List[Document],
    source_files: Optional[List[str]] = None,
    doc_ids: Optional[List[str]] = None,
    page_numbers: Optional[List[int]] = None,
) -> List[Document]:
    out = []

    for d in documents:
        meta = d.metadata or {}
        ok = True

        if source_files and meta.get("source_file") not in source_files:
            ok = False
        if doc_ids and meta.get("doc_id") not in doc_ids:
            ok = False
        if page_numbers and meta.get("page_number") not in page_numbers:
            ok = False

        if ok:
            out.append(d)

    return out


def filter_nodes(
    nodes,
    source_files: Optional[List[str]] = None,
    doc_ids: Optional[List[str]] = None,
    page_numbers: Optional[List[int]] = None,
):
    out = []

    for n in nodes:
        meta = n.metadata or {}
        ok = True

        if source_files and meta.get("source_file") not in source_files:
            ok = False
        if doc_ids and meta.get("doc_id") not in doc_ids:
            ok = False
        if page_numbers and meta.get("page_number") not in page_numbers:
            ok = False

        if ok:
            out.append(n)

    return out


# =============================================================================
# Validation-style answer formatting
# =============================================================================

def build_validation_answer_template(
    question: str,
    raw_answer: str,
    citations: List[Dict[str, Any]],
) -> str:
    key_findings = []
    source_pages = []

    for c in citations[:5]:
        label = c.get("citation_label") or f"{c.get('source_file')} p.{c.get('page_number')}"
        source_pages.append(f"- {label}")
        preview = c.get("text_preview") or ""
        if preview:
            key_findings.append(f"- {preview}")

    if not key_findings:
        key_findings = ["- No explicit evidence snippets returned."]

    if not source_pages:
        source_pages = ["- No page-level source nodes returned."]

    potential_gaps = [
        "- Check whether the retrieved sources are sufficient for a formal validation conclusion.",
        "- Confirm whether there are missing appendices, annexes, or related model-specific documents.",
        "- Review whether the answer covers governance, controls, assumptions, metrics, monitoring, and approvals if relevant.",
    ]

    evidence_lines = []
    for c in citations[:5]:
        evidence_lines.append(
            f"- [{c['rank']}] {c['citation_label']} | score={c['score']}"
        )
    if not evidence_lines:
        evidence_lines = ["- No ranked evidence returned."]

    text = (
        f"Validation View\n"
        f"{raw_answer.strip()}\n\n"
        f"Key Findings\n"
        f"{chr(10).join(key_findings)}\n\n"
        f"Evidence\n"
        f"{chr(10).join(evidence_lines)}\n\n"
        f"Potential Gaps / Follow-up\n"
        f"{chr(10).join(potential_gaps)}\n\n"
        f"Source Pages\n"
        f"{chr(10).join(source_pages)}"
    )
    return text


# =============================================================================
# KB
# =============================================================================

class ValidationRoutedKnowledgeBaseEN:
    def __init__(self):
        self.documents = None
        self.window_nodes = None
        self.standard_nodes = None
        self.vector_index = None

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

    def build(self, documents: List[Document]):
        self.configure_models()
        self.documents = documents

        print("[KB] Build sentence-window nodes...")
        self.window_nodes = build_sentence_window_nodes(documents)
        print(f"[KB] sentence-window nodes = {len(self.window_nodes)}")

        print("[KB] Build standard nodes...")
        self.standard_nodes = build_standard_nodes(documents)
        print(f"[KB] standard nodes = {len(self.standard_nodes)}")

        print("[KB] Build persistent Chroma vector store...")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        chroma_collection = chroma_client.get_or_create_collection("validation_routed_kb_collection_en")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.vector_index = VectorStoreIndex(
            self.window_nodes,
            storage_context=storage_context,
        )

    def _build_postprocessors(self, use_window: bool = True, use_reranker: bool = True):
        postprocessors = []

        if use_window:
            postprocessors.append(
                MetadataReplacementPostProcessor(target_metadata_key="window")
            )

        postprocessors.append(
            SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
        )

        if use_reranker and USE_RERANKER:
            postprocessors.append(
                LLMRerank(top_n=RERANK_TOP_N, llm=Settings.llm)
            )

        return postprocessors

    def _build_detail_engine(
        self,
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
    ):
        metadata_filters = build_metadata_filters(
            source_files=source_files,
            doc_ids=doc_ids,
            page_numbers=page_numbers,
        )

        vector_retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=VECTOR_TOP_K,
            filters=metadata_filters,
        )

        bm25_nodes = filter_nodes(
            self.standard_nodes,
            source_files=source_files,
            doc_ids=doc_ids,
            page_numbers=page_numbers,
        )

        bm25_retriever = BM25Retriever.from_defaults(
            nodes=bm25_nodes,
            similarity_top_k=BM25_TOP_K,
            tokenizer=english_tokenizer,
        )

        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=FUSION_TOP_K,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=True,
        )

        engine = RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            response_mode=DETAIL_RESPONSE_MODE,
            node_postprocessors=self._build_postprocessors(
                use_window=True,
                use_reranker=True,
            ),
        )
        return engine

    def _build_summary_engine(
        self,
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
    ):
        filtered_docs = filter_documents(
            self.documents,
            source_files=source_files,
            doc_ids=doc_ids,
            page_numbers=page_numbers,
        )

        if not filtered_docs:
            filtered_docs = self.documents

        summary_index = SummaryIndex.from_documents(filtered_docs)
        summary_engine = summary_index.as_query_engine(
            response_mode=SUMMARY_RESPONSE_MODE
        )
        return summary_engine

    def _build_router_engine(
        self,
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
    ):
        detail_engine = self._build_detail_engine(
            source_files=source_files,
            doc_ids=doc_ids,
            page_numbers=page_numbers,
        )
        summary_engine = self._build_summary_engine(
            source_files=source_files,
            doc_ids=doc_ids,
            page_numbers=page_numbers,
        )

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_engine,
            description=(
                "Use this tool for validation-style summary questions. "
                "Examples: summarize the document set, explain the overall governance framework, "
                "describe high-level responsibilities, summarize model lifecycle requirements, "
                "provide a reviewer-friendly overview, or synthesize themes across documents."
            ),
        )

        detail_tool = QueryEngineTool.from_defaults(
            query_engine=detail_engine,
            description=(
                "Use this tool for validation-style evidence questions. "
                "Examples: identify exact source pages, retrieve specific clauses, thresholds, "
                "controls, monitoring requirements, approvals, assumptions, limitations, "
                "or any question requiring citation-friendly, page-grounded evidence."
            ),
        )

        router_engine = RouterQueryEngine.from_defaults(
            query_engine_tools=[summary_tool, detail_tool]
        )
        return router_engine

    def answer(
        self,
        question: str,
        mode: str = "router",  # router | summary | detail
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:

        if mode == "summary":
            engine = self._build_summary_engine(
                source_files=source_files,
                doc_ids=doc_ids,
                page_numbers=page_numbers,
            )
        elif mode == "detail":
            engine = self._build_detail_engine(
                source_files=source_files,
                doc_ids=doc_ids,
                page_numbers=page_numbers,
            )
        elif mode == "router":
            engine = self._build_router_engine(
                source_files=source_files,
                doc_ids=doc_ids,
                page_numbers=page_numbers,
            )
        else:
            raise ValueError("mode must be one of: router, summary, detail")

        response = engine.query(question)

        citations = []
        for i, sn in enumerate(getattr(response, "source_nodes", []) or [], start=1):
            node = sn.node
            meta = node.metadata or {}

            citation_label = meta.get("citation_label") or f"{meta.get('source_file')} p.{meta.get('page_number')}"
            preview = meta.get("window") or meta.get("original_text") or node.text

            citations.append({
                "rank": i,
                "score": getattr(sn, "score", None),
                "source_file": meta.get("source_file"),
                "page_number": meta.get("page_number"),
                "doc_id": meta.get("doc_id"),
                "citation_label": citation_label,
                "retrieval_style": meta.get("retrieval_style"),
                "window": meta.get("window"),
                "original_text": meta.get("original_text"),
                "text_preview": short_preview(preview, 400),
            })

        pretty_answer = build_validation_answer_template(
            question=question,
            raw_answer=str(response),
            citations=citations,
        )

        return {
            "question": question,
            "mode": mode,
            "filters": {
                "source_files": source_files,
                "doc_ids": doc_ids,
                "page_numbers": page_numbers,
            },
            "answer": str(response),
            "pretty_answer": pretty_answer,
            "citations": citations,
        }


# =============================================================================
# Output
# =============================================================================

def print_result(result: Dict[str, Any]):
    print("\n" + "=" * 100)
    print(f"QUESTION: {result['question']}")
    print(f"MODE: {result['mode']}")
    print(f"FILTERS: {result['filters']}")
    print("-" * 100)
    print(result["pretty_answer"])


def save_result_json(result: Dict[str, Any], filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    write_json(path, result)
    print(f"[Saved] {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dirs()

    print("[1] Convert all PDFs to page-level markdown with Docling...")
    page_records = convert_folder_pdfs_to_page_markdown(PDF_FOLDER)
    print(f"[1] total page records = {len(page_records)}")

    print("[2] Build Documents with page metadata...")
    documents = page_markdown_records_to_documents(page_records)
    print(f"[2] total documents (pages) = {len(documents)}")

    print("[3] Build Validation Routed Knowledge Base (English)...")
    kb = ValidationRoutedKnowledgeBaseEN()
    kb.build(documents)
    print("[3] knowledge base ready")

    q1 = "Summarize the core governance, risk management, responsibilities, and control requirements across these documents."
    r1 = kb.answer(q1, mode="router")
    print_result(r1)
    save_result_json(r1, "validation_answer_en_01_router_summary_like.json")

    q2 = "Which pages mention responsibilities, approvals, controls, monitoring requirements, or review frequency? Please provide page-grounded evidence."
    r2 = kb.answer(q2, mode="router")
    print_result(r2)
    save_result_json(r2, "validation_answer_en_02_router_detail_like.json")

    q3 = "Summarize the core governance framework in NIST_AI_RMF_1_0.pdf."
    r3 = kb.answer(
        q3,
        mode="summary",
        source_files=["NIST_AI_RMF_1_0.pdf"],
    )
    print_result(r3)
    save_result_json(r3, "validation_answer_en_03_summary_only.json")

    q4 = "Do pages 10 to 12 mention monitoring, metrics, thresholds, reporting, or review frequency?"
    r4 = kb.answer(
        q4,
        mode="detail",
        source_files=["NIST_AI_RMF_1_0.pdf"],
        page_numbers=[10, 11, 12],
    )
    print_result(r4)
    save_result_json(r4, "validation_answer_en_04_detail_filtered_pages.json")


if __name__ == "__main__":
    main()