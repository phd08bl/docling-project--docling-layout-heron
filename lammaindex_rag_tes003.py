import os
import re
import json
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import gradio as gr
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

WORK_DIR = "./kb_app_workspace"
UPLOAD_DIR = os.path.join(WORK_DIR, "uploaded_docs")
PAGE_PDF_DIR = os.path.join(WORK_DIR, "page_pdfs")
PAGE_MD_DIR = os.path.join(WORK_DIR, "page_markdown")
CHROMA_DIR = os.path.join(WORK_DIR, "chroma_db")
MANIFEST_PATH = os.path.join(WORK_DIR, "manifest.json")
THREADS_PATH = os.path.join(WORK_DIR, "threads.json")

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
    for p in [WORK_DIR, UPLOAD_DIR, PAGE_PDF_DIR, PAGE_MD_DIR, CHROMA_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)


def reset_workspace():
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    ensure_dirs()


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
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
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]


def short_preview(text: str, n: int = 320) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:n] + ("..." if len(text) > n else "")


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def normalize_text_for_display(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# =============================================================================
# Thread store
# =============================================================================

class ThreadStore:
    def __init__(self, path: str):
        self.path = path
        self.data = load_json(self.path, {"threads": {}})

    def save(self):
        save_json(self.path, self.data)

    def list_threads(self) -> List[str]:
        return sorted(self.data.get("threads", {}).keys())

    def create_thread(self, name: Optional[str] = None) -> str:
        existing = self.list_threads()
        if name is None:
            i = 1
            while f"Thread {i}" in existing:
                i += 1
            name = f"Thread {i}"
        self.data["threads"][name] = []
        self.save()
        return name

    def delete_thread(self, name: str):
        if name in self.data.get("threads", {}):
            del self.data["threads"][name]
            self.save()

    def rename_thread(self, old_name: str, new_name: str) -> str:
        new_name = (new_name or "").strip()
        if not old_name or old_name not in self.data.get("threads", {}):
            return old_name
        if not new_name:
            return old_name
        if new_name == old_name:
            return old_name
        if new_name in self.data["threads"]:
            raise ValueError("Thread name already exists.")
        self.data["threads"][new_name] = self.data["threads"].pop(old_name)
        self.save()
        return new_name

    def get_messages(self, name: str) -> List[Dict[str, str]]:
        return self.data.get("threads", {}).get(name, [])

    def append_message(self, name: str, role: str, content: str):
        if name not in self.data["threads"]:
            self.data["threads"][name] = []
        self.data["threads"][name].append({"role": role, "content": content})
        self.save()


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
# KB ingestion helpers
# =============================================================================

def convert_page_pdf_to_markdown(converter: DocumentConverter, page_pdf_path: str, md_path: str):
    result = converter.convert(page_pdf_path)
    md_text = result.document.export_to_markdown()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    return md_text


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
# KB
# =============================================================================

class RoutedKnowledgeBaseApp:
    def __init__(self):
        ensure_dirs()
        self.manifest = load_json(MANIFEST_PATH, {})
        self.documents: List[Document] = []
        self.window_nodes = []
        self.standard_nodes = []
        self.vector_index = None
        self.is_ready = False

    def save_manifest(self):
        save_json(MANIFEST_PATH, self.manifest)

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

    def copy_uploaded_files(self, uploaded_files: List[str]) -> Dict[str, Any]:
        ensure_dirs()
        result = {"added": [], "skipped": [], "errors": []}

        if not uploaded_files:
            return result

        existing_hashes = {}
        for existing_name in os.listdir(UPLOAD_DIR):
            existing_path = os.path.join(UPLOAD_DIR, existing_name)
            if os.path.isfile(existing_path):
                try:
                    existing_hashes[file_md5(existing_path)] = existing_name
                except Exception:
                    pass

        for file_obj in uploaded_files:
            try:
                src = file_obj
                if hasattr(file_obj, "name"):
                    src = file_obj.name

                if not src or not os.path.exists(src):
                    result["errors"].append(f"Missing upload file: {src}")
                    continue

                ext = Path(src).suffix.lower()
                if ext not in [".pdf", ".md", ".markdown"]:
                    result["errors"].append(f"Unsupported file type: {os.path.basename(src)}")
                    continue

                md5 = file_md5(src)
                if md5 in existing_hashes:
                    result["skipped"].append(os.path.basename(src))
                    continue

                target_name = os.path.basename(src)
                target_path = os.path.join(UPLOAD_DIR, target_name)

                if os.path.exists(target_path):
                    stem = Path(target_name).stem
                    suffix = Path(target_name).suffix
                    i = 1
                    while os.path.exists(target_path):
                        target_name = f"{stem}_{i}{suffix}"
                        target_path = os.path.join(UPLOAD_DIR, target_name)
                        i += 1

                shutil.copy2(src, target_path)
                result["added"].append(target_name)

            except Exception as e:
                result["errors"].append(f"{os.path.basename(str(file_obj))}: {e}")

        return result

    def delete_document(self, filename: str):
        target = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(target):
            os.remove(target)

    def ingest_all_documents(self, progress: Optional[gr.Progress] = None):
        ensure_dirs()
        self.manifest = {}
        docs: List[Document] = []
        converter = DocumentConverter()

        all_files = sorted(Path(UPLOAD_DIR).glob("*"))
        total_files = max(len(all_files), 1)

        for idx, file_path in enumerate(all_files, start=1):
            if progress:
                progress((idx - 1) / total_files, desc=f"Processing {file_path.name}")

            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            source_file = file_path.name
            source_path = str(file_path)
            source_hash = file_md5(source_path)

            if ext == ".pdf":
                page_pdfs = split_pdf_to_pages(source_path, PAGE_PDF_DIR)
                page_records = []

                for page_number, page_pdf_path in page_pdfs:
                    doc_id = f"{safe_stem(source_path)}__p{page_number:04d}"
                    page_md_path = os.path.join(PAGE_MD_DIR, f"{doc_id}.md")

                    convert_page_pdf_to_markdown(converter, page_pdf_path, page_md_path)
                    with open(page_md_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()

                    if text:
                        metadata = {
                            "source_file": source_file,
                            "source_path": source_path,
                            "page_number": page_number,
                            "doc_id": doc_id,
                            "page_markdown_path": page_md_path,
                            "citation_label": f"{source_file} p.{page_number}",
                            "file_type": "pdf",
                        }
                        docs.append(Document(text=text, metadata=metadata))

                    page_records.append({
                        "page_number": page_number,
                        "page_pdf_path": page_pdf_path,
                        "page_markdown_path": page_md_path,
                        "doc_id": doc_id,
                    })

                self.manifest[source_path] = {
                    "source_file": source_file,
                    "file_type": "pdf",
                    "md5": source_hash,
                    "pages": page_records,
                }

            elif ext in [".md", ".markdown"]:
                with open(source_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                doc_id = safe_stem(source_path)
                if text:
                    metadata = {
                        "source_file": source_file,
                        "source_path": source_path,
                        "page_number": None,
                        "doc_id": doc_id,
                        "citation_label": source_file,
                        "file_type": "markdown",
                    }
                    docs.append(Document(text=text, metadata=metadata))

                self.manifest[source_path] = {
                    "source_file": source_file,
                    "file_type": "markdown",
                    "md5": source_hash,
                    "doc_id": doc_id,
                }

        self.save_manifest()
        self.documents = docs

        if DELETE_TEMP_PAGE_PDFS:
            shutil.rmtree(PAGE_PDF_DIR, ignore_errors=True)
            Path(PAGE_PDF_DIR).mkdir(parents=True, exist_ok=True)

    def build_nodes(self, progress: Optional[gr.Progress] = None):
        if progress:
            progress(0.70, desc="Building sentence-window nodes")

        node_parser = SentenceWindowNodeParser.from_defaults(
            sentence_splitter=english_sentence_splitter,
            window_size=WINDOW_SIZE,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        self.window_nodes = node_parser.get_nodes_from_documents(self.documents)
        for n in self.window_nodes:
            if n.metadata is None:
                n.metadata = {}
            n.metadata["retrieval_style"] = "sentence_window"
            n.excluded_embed_metadata_keys = []
            n.excluded_llm_metadata_keys = []

        if progress:
            progress(0.82, desc="Building standard nodes")

        splitter = SentenceSplitter(
            chunk_size=BASE_CHUNK_SIZE,
            chunk_overlap=BASE_CHUNK_OVERLAP,
        )
        self.standard_nodes = splitter.get_nodes_from_documents(self.documents)
        for n in self.standard_nodes:
            if n.metadata is None:
                n.metadata = {}
            n.metadata["retrieval_style"] = "standard_chunk"
            n.excluded_embed_metadata_keys = []
            n.excluded_llm_metadata_keys = []

    def build_index(self, progress: Optional[gr.Progress] = None):
        self.configure_models()

        if progress:
            progress(0.90, desc="Building vector index")

        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            chroma_client.delete_collection("kb_ui_collection")
        except Exception:
            pass

        chroma_collection = chroma_client.get_or_create_collection("kb_ui_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.vector_index = VectorStoreIndex(
            self.window_nodes,
            storage_context=storage_context,
        )

    def rebuild_kb(self, progress: Optional[gr.Progress] = None):
        self.is_ready = False
        self.ingest_all_documents(progress=progress)

        if not self.documents:
            self.window_nodes = []
            self.standard_nodes = []
            self.vector_index = None
            self.is_ready = False
            return

        self.build_nodes(progress=progress)
        self.build_index(progress=progress)
        self.is_ready = True

        if progress:
            progress(1.0, desc="Knowledge base ready")

    def get_document_table(self) -> List[List[Any]]:
        rows = []
        for source_path, info in sorted(self.manifest.items(), key=lambda x: x[1]["source_file"].lower()):
            file_type = info.get("file_type")
            source_file = info.get("source_file")
            if file_type == "pdf":
                page_count = len(info.get("pages", []))
                rows.append([source_file, "PDF", page_count, source_path])
            else:
                rows.append([source_file, "Markdown", "-", source_path])
        return rows

    def get_document_names(self) -> List[str]:
        return [row[0] for row in self.get_document_table()]

    def clear_all(self):
        reset_workspace()
        self.manifest = {}
        self.documents = []
        self.window_nodes = []
        self.standard_nodes = []
        self.vector_index = None
        self.is_ready = False

    def _build_postprocessors(self):
        postprocessors = [
            MetadataReplacementPostProcessor(target_metadata_key="window"),
            SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF),
        ]
        if USE_RERANKER:
            postprocessors.append(LLMRerank(top_n=RERANK_TOP_N, llm=Settings.llm))
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
        )

        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=FUSION_TOP_K,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=True,
        )

        return RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            response_mode=DETAIL_RESPONSE_MODE,
            node_postprocessors=self._build_postprocessors(),
        )

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
        return summary_index.as_query_engine(response_mode=SUMMARY_RESPONSE_MODE)

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
                "Use this tool for high-level overview questions, summarization, broad themes, "
                "document-wide understanding, executive summaries, or synthesis across one or more documents."
            ),
        )

        detail_tool = QueryEngineTool.from_defaults(
            query_engine=detail_engine,
            description=(
                "Use this tool for specific evidence questions, citation-friendly retrieval, "
                "page-grounded answers, definitions, clauses, thresholds, requirements, responsibilities, "
                "controls, monitoring details, or exact supporting text."
            ),
        )

        return RouterQueryEngine.from_defaults(
            query_engine_tools=[summary_tool, detail_tool]
        )

    def answer(
        self,
        question: str,
        mode: str = "router",
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        if not self.is_ready or self.vector_index is None:
            return {
                "answer": "The knowledge base is empty. Please upload PDF or Markdown documents first.",
                "citations": [],
                "answer_html": "<div class='answer-block'><div class='answer-title'>Answer</div><div>The knowledge base is empty. Please upload PDF or Markdown documents first.</div></div>",
                "sources_html": "<div class='no-sources'>No sources available.</div>",
            }

        if mode == "summary":
            engine = self._build_summary_engine(source_files, doc_ids, page_numbers)
        elif mode == "detail":
            engine = self._build_detail_engine(source_files, doc_ids, page_numbers)
        else:
            engine = self._build_router_engine(source_files, doc_ids, page_numbers)

        response = engine.query(question)

        citations = []
        for i, sn in enumerate(getattr(response, "source_nodes", []) or [], start=1):
            node = sn.node
            meta = node.metadata or {}

            citation_label = meta.get("citation_label")
            if not citation_label:
                if meta.get("page_number") is not None:
                    citation_label = f"{meta.get('source_file')} p.{meta.get('page_number')}"
                else:
                    citation_label = str(meta.get("source_file"))

            preview = meta.get("window") or meta.get("original_text") or node.text
            citations.append({
                "rank": i,
                "score": getattr(sn, "score", None),
                "source_file": meta.get("source_file"),
                "page_number": meta.get("page_number"),
                "doc_id": meta.get("doc_id"),
                "citation_label": citation_label,
                "retrieval_style": meta.get("retrieval_style"),
                "text_preview": short_preview(preview, 700),
            })

        answer_html = self._format_answer_html(str(response), citations)
        sources_html = self._format_sources_panel_html(citations)

        return {
            "answer": str(response),
            "citations": citations,
            "answer_html": answer_html,
            "sources_html": sources_html,
        }

    def _format_answer_html(self, raw_answer: str, citations: List[Dict[str, Any]]) -> str:
        badges = ""
        for c in citations[:8]:
            badges += (
                f"<a class='citation-badge' href='#source-card-{c['rank']}' "
                f"onclick=\"setTimeout(function(){{"
                f"var el=document.getElementById('source-card-{c['rank']}');"
                f"if(el){{el.open=true;el.scrollIntoView({{behavior:'smooth', block:'center'}});}}"
                f"}},50);\">"
                f"{html_escape(c['citation_label'])}</a>"
            )

        return f"""
        <div class="answer-block">
          <div class="answer-title">Answer</div>
          <div class="answer-text">{html_escape(raw_answer)}</div>
          <div class="citation-badges">{badges}</div>
        </div>
        """

    def _format_sources_panel_html(self, citations: List[Dict[str, Any]]) -> str:
        if not citations:
            return "<div class='no-sources'>No page-level source nodes returned.</div>"

        cards = ""
        for c in citations:
            score = c.get("score")
            score_text = f"{score:.4f}" if isinstance(score, (float, int)) else str(score)
            cards += f"""
            <details class="source-card" id="source-card-{c['rank']}">
              <summary>
                <span class="source-rank">[{c['rank']}]</span>
                <span class="source-label">{html_escape(c['citation_label'])}</span>
                <span class="source-score">score={html_escape(score_text)}</span>
              </summary>
              <div class="source-body">
                <div class="source-meta"><b>Retrieval:</b> {html_escape(str(c.get('retrieval_style')))}</div>
                <div class="source-preview">{html_escape(c['text_preview'])}</div>
              </div>
            </details>
            """
        return f"<div class='sources-panel-wrapper'>{cards}</div>"


# =============================================================================
# App instances
# =============================================================================

KB = RoutedKnowledgeBaseApp()
THREADS = ThreadStore(THREADS_PATH)

if not THREADS.list_threads():
    THREADS.create_thread("Thread 1")


# =============================================================================
# UI helpers
# =============================================================================

def build_thread_sidebar_html(current_thread: Optional[str]) -> str:
    names = THREADS.list_threads()
    if not names:
        return "<div class='thread-sidebar-empty'>No threads</div>"

    html = "<div class='thread-sidebar-list'>"
    for name in names:
        active_cls = "thread-sidebar-item active" if name == current_thread else "thread-sidebar-item"
        html += f"<div class='{active_cls}'>{html_escape(name)}</div>"
    html += "</div>"
    return html


def ui_refresh_documents(progress=gr.Progress()):
    KB.rebuild_kb(progress=progress)
    rows = KB.get_document_table()
    choices = KB.get_document_names()
    msg = f"Knowledge base refreshed. Documents loaded: {len(rows)}"
    return rows, choices, msg


def ui_add_documents(files, progress=gr.Progress()):
    if not files:
        return KB.get_document_table(), KB.get_document_names(), "No files selected."

    copied = KB.copy_uploaded_files(files)
    KB.rebuild_kb(progress=progress)

    parts = []
    if copied["added"]:
        parts.append(f"Added: {', '.join(copied['added'])}")
    if copied["skipped"]:
        parts.append(f"Skipped duplicates: {', '.join(copied['skipped'])}")
    if copied["errors"]:
        parts.append(f"Errors: {' | '.join(copied['errors'])}")
    if not parts:
        parts.append("No changes.")

    return KB.get_document_table(), KB.get_document_names(), "\n".join(parts)


def ui_delete_selected_document(selected_doc_name, progress=gr.Progress()):
    if not selected_doc_name:
        return KB.get_document_table(), KB.get_document_names(), "No document selected."

    KB.delete_document(selected_doc_name)
    KB.rebuild_kb(progress=progress)
    return KB.get_document_table(), KB.get_document_names(), f"Deleted: {selected_doc_name}"


def ui_clear_all_documents():
    KB.clear_all()
    THREADS.data = {"threads": {}}
    THREADS.create_thread("Thread 1")
    THREADS.save()
    current = "Thread 1"
    return (
        [],
        [],
        "All documents and threads cleared.",
        gr.update(choices=THREADS.list_threads(), value=current),
        build_thread_sidebar_html(current),
        [],
        "",
        "<div class='no-sources'>No sources yet.</div>",
    )


def ui_get_threads():
    names = THREADS.list_threads()
    if not names:
        name = THREADS.create_thread("Thread 1")
        names = [name]
    return names


def ui_load_thread(thread_name):
    if not thread_name:
        return [], "", build_thread_sidebar_html(None), "<div class='no-sources'>No sources yet.</div>"

    messages = THREADS.get_messages(thread_name)

    # load latest assistant sources if available
    latest_sources = "<div class='no-sources'>No sources yet.</div>"
    for msg in reversed(messages):
        if msg["role"] == "assistant_sources":
            latest_sources = msg["content"]
            break

    display_messages = []
    for msg in messages:
        if msg["role"] == "assistant_sources":
            continue
        role = "assistant" if msg["role"] == "assistant" else "user"
        display_messages.append({"role": role, "content": msg["content"]})

    return display_messages, "", build_thread_sidebar_html(thread_name), latest_sources


def ui_new_thread():
    name = THREADS.create_thread()
    choices = THREADS.list_threads()
    return (
        gr.update(choices=choices, value=name),
        [],
        "",
        build_thread_sidebar_html(name),
        "<div class='no-sources'>No sources yet.</div>",
    )


def ui_delete_thread(thread_name):
    if thread_name:
        THREADS.delete_thread(thread_name)

    names = THREADS.list_threads()
    if not names:
        new_name = THREADS.create_thread("Thread 1")
        names = [new_name]

    current = names[0]
    messages = THREADS.get_messages(current)
    display_messages = [{"role": ("assistant" if m["role"] == "assistant" else "user"), "content": m["content"]}
                        for m in messages if m["role"] != "assistant_sources"]

    latest_sources = "<div class='no-sources'>No sources yet.</div>"
    for msg in reversed(messages):
        if msg["role"] == "assistant_sources":
            latest_sources = msg["content"]
            break

    return (
        gr.update(choices=names, value=current),
        display_messages,
        "",
        build_thread_sidebar_html(current),
        latest_sources,
    )


def ui_rename_thread(current_name, new_name):
    try:
        renamed = THREADS.rename_thread(current_name, new_name)
        choices = THREADS.list_threads()
        messages = THREADS.get_messages(renamed)
        display_messages = [{"role": ("assistant" if m["role"] == "assistant" else "user"), "content": m["content"]}
                            for m in messages if m["role"] != "assistant_sources"]

        latest_sources = "<div class='no-sources'>No sources yet.</div>"
        for msg in reversed(messages):
            if msg["role"] == "assistant_sources":
                latest_sources = msg["content"]
                break

        return (
            gr.update(choices=choices, value=renamed),
            display_messages,
            f"Renamed to: {renamed}",
            "",
            build_thread_sidebar_html(renamed),
            latest_sources,
        )
    except Exception as e:
        return gr.update(), gr.update(), f"Rename failed: {e}", new_name, gr.update(), gr.update()


def ui_send_message(thread_name, message, chat_history, mode, selected_sources, progress=gr.Progress()):
    if not thread_name:
        thread_name = THREADS.create_thread("Thread 1")

    if not message or not message.strip():
        latest_sources = "<div class='no-sources'>No sources yet.</div>"
        return chat_history, "", gr.update(value=thread_name), "", build_thread_sidebar_html(thread_name), latest_sources

    user_msg = message.strip()
    THREADS.append_message(thread_name, "user", user_msg)

    progress(0.2, desc="Searching knowledge base")
    source_files = selected_sources if selected_sources else None
    result = KB.answer(user_msg, mode=mode, source_files=source_files)

    progress(0.85, desc="Formatting answer")
    assistant_html = result["answer_html"]
    sources_html = result["sources_html"]

    THREADS.append_message(thread_name, "assistant", assistant_html)
    THREADS.append_message(thread_name, "assistant_sources", sources_html)

    messages = THREADS.get_messages(thread_name)
    display_messages = [{"role": ("assistant" if m["role"] == "assistant" else "user"), "content": m["content"]}
                        for m in messages if m["role"] != "assistant_sources"]

    progress(1.0, desc="Done")
    return (
        display_messages,
        "",
        gr.update(value=thread_name),
        "",
        build_thread_sidebar_html(thread_name),
        sources_html,
    )


# =============================================================================
# Build UI
# =============================================================================

CUSTOM_CSS = """
#main-title {
  font-size: 34px;
  font-weight: 700;
  margin-bottom: 6px;
}
.orange-btn button {
  background: linear-gradient(90deg, #f97316, #ea580c) !important;
  color: white !important;
  border: none !important;
}
.red-btn button {
  background: linear-gradient(90deg, #ef4444, #e11d48) !important;
  color: white !important;
  border: none !important;
}
.soft-btn button {
  background: #e5e7eb !important;
  color: #111827 !important;
  border: none !important;
}
.answer-block {
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 14px;
  background: #ffffff;
}
.answer-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 10px;
}
.answer-text {
  white-space: pre-wrap;
  line-height: 1.6;
  margin-bottom: 12px;
}
.citation-badges {
  margin-bottom: 12px;
}
.citation-badge {
  display: inline-block;
  margin-right: 8px;
  margin-bottom: 8px;
  padding: 4px 10px;
  border-radius: 999px;
  background: #fff7ed;
  border: 1px solid #fdba74;
  color: #9a3412;
  font-size: 12px;
  font-weight: 600;
  text-decoration: none !important;
}
.citation-badge:hover {
  background: #ffedd5;
}
.sources-title {
  font-weight: 700;
  margin-bottom: 10px;
}
.source-card {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 8px 12px;
  margin-bottom: 10px;
  background: #fafafa;
}
.source-card summary {
  cursor: pointer;
  list-style: none;
  display: flex;
  gap: 10px;
  align-items: center;
  font-weight: 600;
}
.source-card summary::-webkit-details-marker {
  display: none;
}
.source-rank {
  color: #c2410c;
}
.source-label {
  flex: 1;
}
.source-score {
  color: #6b7280;
  font-size: 12px;
}
.source-body {
  margin-top: 10px;
}
.source-preview {
  white-space: pre-wrap;
  line-height: 1.5;
  color: #111827;
}
.no-sources {
  color: #6b7280;
}
.footer-hint {
  text-align: center;
  color: #6b7280;
  margin-top: 18px;
}
.thread-sidebar-wrapper {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  background: #fafafa;
  min-height: 500px;
  padding: 10px;
}
.thread-sidebar-title {
  font-weight: 700;
  margin-bottom: 10px;
}
.thread-sidebar-item {
  padding: 10px 12px;
  border-radius: 10px;
  background: white;
  margin-bottom: 8px;
  border: 1px solid #e5e7eb;
}
.thread-sidebar-item.active {
  border-color: #fdba74;
  background: #fff7ed;
}
.right-panel-wrapper {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  background: #fafafa;
  min-height: 500px;
  padding: 10px;
}
.right-panel-title {
  font-weight: 700;
  margin-bottom: 10px;
}
"""


def build_ui():
    with gr.Blocks(css=CUSTOM_CSS, title="Agentic RAG Chatbot with Cortex AI") as demo:
        gr.Markdown("# 🤖 Agentic RAG Chatbot with Cortex AI", elem_id="main-title")

        with gr.Tabs():
            with gr.Tab("Documents"):
                gr.Markdown("## Add New Documents")
                gr.Markdown("Upload PDF or Markdown files. Duplicates will be automatically skipped.")

                file_input = gr.File(
                    label="File",
                    file_count="multiple",
                    file_types=[".pdf", ".md", ".markdown"],
                )

                add_btn = gr.Button("Add Documents", elem_classes=["orange-btn"])
                doc_status = gr.Textbox(label="Status", interactive=False, lines=4)

                gr.Markdown("## Current Uploaded Documents in the Knowledge Base")
                doc_table = gr.Dataframe(
                    headers=["Document", "Type", "Pages", "Path"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    value=KB.get_document_table(),
                )

                with gr.Row():
                    doc_delete_dropdown = gr.Dropdown(
                        label="Select document to delete",
                        choices=KB.get_document_names(),
                        interactive=True,
                    )
                    delete_doc_btn = gr.Button("Delete Selected Document", elem_classes=["red-btn"])

                with gr.Row():
                    refresh_btn = gr.Button("Refresh", elem_classes=["soft-btn"])
                    clear_btn = gr.Button("Clear All", elem_classes=["red-btn"])

            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=220):
                        gr.HTML("<div class='thread-sidebar-title'>Threads</div>")
                        thread_sidebar = gr.HTML(build_thread_sidebar_html(ui_get_threads()[0]), elem_classes=["thread-sidebar-wrapper"])

                    with gr.Column(scale=7, min_width=500):
                        with gr.Row():
                            thread_dropdown = gr.Dropdown(
                                label="Current Thread",
                                choices=ui_get_threads(),
                                value=ui_get_threads()[0],
                                interactive=True,
                            )
                            new_thread_btn = gr.Button("New Thread", elem_classes=["orange-btn"])
                            delete_thread_btn = gr.Button("Delete Thread", elem_classes=["red-btn"])

                        with gr.Row():
                            rename_box = gr.Textbox(label="Rename current thread", placeholder="Enter new thread name")
                            rename_thread_btn = gr.Button("Rename Thread", elem_classes=["soft-btn"])
                            thread_status = gr.Textbox(label="Status", interactive=False, lines=1)

                        with gr.Row():
                            mode_selector = gr.Radio(
                                choices=["router", "summary", "detail"],
                                value="router",
                                label="Mode",
                            )
                            source_filter = gr.Dropdown(
                                label="Filter by source documents (optional)",
                                choices=KB.get_document_names(),
                                multiselect=True,
                                interactive=True,
                            )

                        chatbot = gr.Chatbot(
                            label="Chatbot",
                            value=THREADS.get_messages(ui_get_threads()[0]),
                            type="messages",
                            height=520,
                            show_copy_button=True,
                        )

                        gr.Markdown(
                            "<div class='footer-hint'><b>Ask me anything!</b><br/>"
                            "I'll search, reason, and act to give you the best answer :)</div>"
                        )

                        message_box = gr.Textbox(
                            label="Message",
                            placeholder="Type a message...",
                            lines=3,
                        )

                        send_btn = gr.Button("Send", elem_classes=["orange-btn"])

                    with gr.Column(scale=4, min_width=320):
                        gr.HTML("<div class='right-panel-title'>Sources</div>")
                        sources_panel = gr.HTML(
                            "<div class='no-sources'>No sources yet.</div>",
                            elem_classes=["right-panel-wrapper"],
                        )

        # Documents actions
        add_btn.click(
            fn=ui_add_documents,
            inputs=[file_input],
            outputs=[doc_table, doc_delete_dropdown, doc_status],
            show_progress="full",
        )

        refresh_btn.click(
            fn=ui_refresh_documents,
            inputs=[],
            outputs=[doc_table, doc_delete_dropdown, doc_status],
            show_progress="full",
        )

        delete_doc_btn.click(
            fn=ui_delete_selected_document,
            inputs=[doc_delete_dropdown],
            outputs=[doc_table, doc_delete_dropdown, doc_status],
            show_progress="full",
        )

        clear_btn.click(
            fn=ui_clear_all_documents,
            inputs=[],
            outputs=[doc_table, doc_delete_dropdown, doc_status, thread_dropdown, thread_sidebar, chatbot, thread_status, sources_panel],
        )

        # Chat actions
        thread_dropdown.change(
            fn=ui_load_thread,
            inputs=[thread_dropdown],
            outputs=[chatbot, thread_status, thread_sidebar, sources_panel],
        )

        new_thread_btn.click(
            fn=ui_new_thread,
            inputs=[],
            outputs=[thread_dropdown, chatbot, thread_status, thread_sidebar, sources_panel],
        )

        delete_thread_btn.click(
            fn=ui_delete_thread,
            inputs=[thread_dropdown],
            outputs=[thread_dropdown, chatbot, thread_status, thread_sidebar, sources_panel],
        )

        rename_thread_btn.click(
            fn=ui_rename_thread,
            inputs=[thread_dropdown, rename_box],
            outputs=[thread_dropdown, chatbot, thread_status, rename_box, thread_sidebar, sources_panel],
        )

        send_btn.click(
            fn=ui_send_message,
            inputs=[thread_dropdown, message_box, chatbot, mode_selector, source_filter],
            outputs=[chatbot, message_box, thread_dropdown, thread_status, thread_sidebar, sources_panel],
            show_progress="full",
        )

        message_box.submit(
            fn=ui_send_message,
            inputs=[thread_dropdown, message_box, chatbot, mode_selector, source_filter],
            outputs=[chatbot, message_box, thread_dropdown, thread_status, thread_sidebar, sources_panel],
            show_progress="full",
        )

    return demo


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    ensure_dirs()
    KB.rebuild_kb()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)