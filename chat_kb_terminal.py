import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import chromadb

from llama_index.core import Document, Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SimilarityPostprocessor, LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore


# =============================================================================
# Config
# =============================================================================

WORK_DIR = "./kb_terminal_workspace"
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

VECTOR_TOP_K = 6
BM25_TOP_K = 6
FUSION_TOP_K = 6
SIMILARITY_CUTOFF = 0.15
MAX_DISPLAY_CITATIONS = 5

USE_RERANKER = False
RERANK_TOP_N = 3
RERANK_MIN_FUSION_CANDIDATES = 8
MAX_SECTION_DOC_CHARS = 3200

DETAIL_RESPONSE_MODE = "compact"
SUMMARY_RESPONSE_MODE = "tree_summarize"


def log(message: str):
    print(f"[chat_kb_terminal] {message}", flush=True)


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def english_sentence_splitter(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def short_preview(text: str, length: int = 320) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:length] + ("..." if len(text) > length else "")


def normalize_text_for_display(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def text_matches_marker(text: str, marker: str) -> bool:
    pattern = r"\b" + re.escape(marker) + r"\b"
    return re.search(pattern, text) is not None


def safe_meta_value(value: Optional[str]) -> str:
    value = normalize_text_for_display(str(value or ""))
    return value if value else "unknown"


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


def build_metadata_filters(
    source_files: Optional[List[str]] = None,
    doc_ids: Optional[List[str]] = None,
    page_numbers: Optional[List[int]] = None,
    document_types: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    policy_names: Optional[List[str]] = None,
) -> Optional[MetadataFilters]:
    filters = []
    if source_files:
        for value in source_files:
            filters.append(MetadataFilter(key="source_file", value=value, operator=FilterOperator.EQ))
    if doc_ids:
        for value in doc_ids:
            filters.append(MetadataFilter(key="doc_id", value=value, operator=FilterOperator.EQ))
    if page_numbers:
        for value in page_numbers:
            filters.append(MetadataFilter(key="page_number", value=value, operator=FilterOperator.EQ))
    if document_types:
        for value in document_types:
            filters.append(MetadataFilter(key="document_type", value=value, operator=FilterOperator.EQ))
    if model_names:
        for value in model_names:
            filters.append(MetadataFilter(key="model_name", value=value, operator=FilterOperator.EQ))
    if policy_names:
        for value in policy_names:
            filters.append(MetadataFilter(key="policy_name", value=value, operator=FilterOperator.EQ))
    if not filters:
        return None
    return MetadataFilters(filters=filters)


def filter_documents(
    documents: List[Document],
    source_files: Optional[List[str]] = None,
    doc_ids: Optional[List[str]] = None,
    page_numbers: Optional[List[int]] = None,
    document_types: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    policy_names: Optional[List[str]] = None,
) -> List[Document]:
    filtered = []
    for document in documents:
        meta = document.metadata or {}
        ok = True
        if source_files and meta.get("source_file") not in source_files:
            ok = False
        if doc_ids and meta.get("doc_id") not in doc_ids:
            ok = False
        if page_numbers and meta.get("page_number") not in page_numbers:
            ok = False
        if document_types and safe_meta_value(meta.get("document_type")) not in document_types:
            ok = False
        if model_names and safe_meta_value(meta.get("model_name")) not in model_names:
            ok = False
        if policy_names and safe_meta_value(meta.get("policy_name")) not in policy_names:
            ok = False
        if ok:
            filtered.append(document)
    return filtered


def filter_nodes(
    nodes,
    source_files: Optional[List[str]] = None,
    doc_ids: Optional[List[str]] = None,
    page_numbers: Optional[List[int]] = None,
    document_types: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    policy_names: Optional[List[str]] = None,
):
    filtered = []
    for node in nodes:
        meta = node.metadata or {}
        ok = True
        if source_files and meta.get("source_file") not in source_files:
            ok = False
        if doc_ids and meta.get("doc_id") not in doc_ids:
            ok = False
        if page_numbers and meta.get("page_number") not in page_numbers:
            ok = False
        if document_types and safe_meta_value(meta.get("document_type")) not in document_types:
            ok = False
        if model_names and safe_meta_value(meta.get("model_name")) not in model_names:
            ok = False
        if policy_names and safe_meta_value(meta.get("policy_name")) not in policy_names:
            ok = False
        if ok:
            filtered.append(node)
    return filtered


def format_sources_text(citations: List[Dict[str, Any]]) -> str:
    if not citations:
        return "No page-level source nodes returned."
    lines = []
    for citation in citations:
        lines.append(
            f"[{citation['rank']}] {citation['citation_label']} | "
            f"section={citation.get('section_title')} | "
            f"retrieval={citation.get('retrieval_style')} | "
            f"score={citation.get('score')}"
        )
        lines.append(f"    preview: {citation.get('text_preview')}")
    return "\n".join(lines)


def is_follow_up_question(question: str) -> bool:
    lowered = f" {question.lower()} "
    markers = [" it ", " they ", " them ", " that ", " those ", " this ", " these ", " earlier ", " previous ", " above ", " same "]
    return any(marker in lowered for marker in markers)


def build_conversational_question(question: str, history: List[Dict[str, str]], max_turns: int = 2) -> str:
    if not history or not is_follow_up_question(question):
        return question
    recent_history = history[-max_turns * 2:]
    context_lines = ["Conversation context:"]
    for item in recent_history:
        context_lines.append(f"{item['role'].capitalize()}: {item['content']}")
    context_lines.append(f"Current user question: {question}")
    return "\n".join(context_lines)


class TerminalRoutedKnowledgeBase:
    def __init__(self):
        self.manifest = load_json(MANIFEST_PATH, {})
        self.stats = load_json(KB_STATS_PATH, {})
        self.documents: List[Document] = []
        self.summary_documents: List[Document] = []
        self.window_nodes = []
        self.standard_nodes = []
        self.vector_index = None
        self.is_ready = False
        self.detail_engine_default = None
        self.detail_retriever_default = None
        self.summary_engine_full = None
        self.router_engine_default = None
        self.bm25_retriever_default = None
        self.postprocessors_default = None

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

    def _reset_query_cache(self):
        self.detail_engine_default = None
        self.detail_retriever_default = None
        self.summary_engine_full = None
        self.router_engine_default = None
        self.bm25_retriever_default = None
        self.postprocessors_default = None

    def _has_active_filters(
        self,
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        document_types: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        policy_names: Optional[List[str]] = None,
    ) -> bool:
        return bool(source_files or doc_ids or page_numbers or document_types or model_names or policy_names)

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
                "citation_label": (
                    f"{base_metadata.get('source_file')} p.{page_number} - {section.get('section_title', 'Section')}"
                    if page_number is not None
                    else f"{base_metadata.get('source_file')} - {section.get('section_title', 'Section')}"
                ),
                "retrieval_style": "section_summary",
            })
            section_docs.append(Document(text=section_text, metadata=metadata))
        return section_docs

    def _load_documents_from_manifest(self) -> List[Document]:
        docs: List[Document] = []
        summary_docs: List[Document] = []
        for source_path, info in sorted(self.manifest.items(), key=lambda item: item[1]["source_file"].lower()):
            source_file = info.get("source_file")
            base_metadata = info.get("base_metadata", {})

            for page in info.get("pages", []):
                page_md_path = page.get("page_markdown_path")
                if not page_md_path or not os.path.exists(page_md_path):
                    raise FileNotFoundError(f"Missing cached markdown page: {page_md_path}")

                with open(page_md_path, "r", encoding="utf-8") as handle:
                    text = handle.read().strip()
                if not text:
                    continue

                page_number = page.get("page_number")
                doc_id = page.get("doc_id")
                metadata = dict(base_metadata)
                metadata.update(page.get("metadata", {}))
                metadata.update({
                    "source_file": source_file,
                    "source_path": source_path,
                    "page_number": page_number,
                    "doc_id": doc_id,
                    "page_markdown_path": page_md_path,
                    "citation_label": page.get("metadata", {}).get("citation_label", f"{source_file} p.{page_number}"),
                    "file_type": "pdf",
                })
                docs.append(Document(text=text, metadata=metadata))
                summary_docs.extend(self._build_section_documents(text, metadata, page_number=page_number))

        self.summary_documents = summary_docs or docs[:]
        return docs

    def build_nodes(self):
        log("loading/rebuilding retrievers/query engines: rebuilding sentence-window and standard nodes")
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

        splitter = SentenceSplitter(chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_CHUNK_OVERLAP)
        self.standard_nodes = splitter.get_nodes_from_documents(self.documents)
        for node in self.standard_nodes:
            if node.metadata is None:
                node.metadata = {}
            node.metadata["retrieval_style"] = "standard_chunk"
            node.excluded_embed_metadata_keys = []
            node.excluded_llm_metadata_keys = []

    def _build_postprocessors(self, candidate_pool_size: int = FUSION_TOP_K):
        postprocessors = [
            MetadataReplacementPostProcessor(target_metadata_key="window"),
            SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF),
        ]
        if USE_RERANKER and candidate_pool_size >= RERANK_MIN_FUSION_CANDIDATES:
            postprocessors.append(LLMRerank(top_n=RERANK_TOP_N, llm=Settings.llm))
        return postprocessors

    def _build_detail_retriever(
        self,
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        document_types: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        policy_names: Optional[List[str]] = None,
    ):
        if not self._has_active_filters(source_files, doc_ids, page_numbers, document_types, model_names, policy_names) and self.detail_retriever_default is not None:
            return self.detail_retriever_default

        metadata_filters = build_metadata_filters(
            source_files=source_files,
            doc_ids=doc_ids,
            page_numbers=page_numbers,
            document_types=document_types,
            model_names=model_names,
            policy_names=policy_names,
        )
        vector_retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=VECTOR_TOP_K,
            filters=metadata_filters,
        )

        if not self._has_active_filters(source_files, doc_ids, page_numbers, document_types, model_names, policy_names) and self.bm25_retriever_default is not None:
            bm25_retriever = self.bm25_retriever_default
        else:
            bm25_nodes = filter_nodes(
                self.standard_nodes,
                source_files=source_files,
                doc_ids=doc_ids,
                page_numbers=page_numbers,
                document_types=document_types,
                model_names=model_names,
                policy_names=policy_names,
            )
            bm25_retriever = BM25Retriever.from_defaults(nodes=bm25_nodes, similarity_top_k=BM25_TOP_K)

        return QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=FUSION_TOP_K,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )

    def _build_detail_engine(
        self,
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        document_types: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        policy_names: Optional[List[str]] = None,
    ):
        if not self._has_active_filters(source_files, doc_ids, page_numbers, document_types, model_names, policy_names) and self.detail_engine_default is not None:
            return self.detail_engine_default

        fusion_retriever = self._build_detail_retriever(
            source_files=source_files,
            doc_ids=doc_ids,
            page_numbers=page_numbers,
            document_types=document_types,
            model_names=model_names,
            policy_names=policy_names,
        )
        return RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            response_mode=DETAIL_RESPONSE_MODE,
            node_postprocessors=self.postprocessors_default
            if not self._has_active_filters(source_files, doc_ids, page_numbers, document_types, model_names, policy_names) and self.postprocessors_default is not None
            else self._build_postprocessors(candidate_pool_size=FUSION_TOP_K),
        )

    def _build_summary_engine(
        self,
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        document_types: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        policy_names: Optional[List[str]] = None,
    ):
        if not self._has_active_filters(source_files, doc_ids, page_numbers, document_types, model_names, policy_names) and self.summary_engine_full is not None:
            return self.summary_engine_full

        filtered_docs = filter_documents(
            self.summary_documents or self.documents,
            source_files=source_files,
            doc_ids=doc_ids,
            page_numbers=page_numbers,
            document_types=document_types,
            model_names=model_names,
            policy_names=policy_names,
        )
        if not filtered_docs:
            filtered_docs = self.summary_documents or self.documents

        summary_index = SummaryIndex.from_documents(filtered_docs)
        return summary_index.as_query_engine(response_mode=SUMMARY_RESPONSE_MODE)

    def _build_router_engine(
        self,
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        document_types: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        policy_names: Optional[List[str]] = None,
    ):
        if not self._has_active_filters(source_files, doc_ids, page_numbers, document_types, model_names, policy_names) and self.router_engine_default is not None:
            return self.router_engine_default

        detail_engine = self._build_detail_engine(source_files, doc_ids, page_numbers, document_types, model_names, policy_names)
        summary_engine = self._build_summary_engine(source_files, doc_ids, page_numbers, document_types, model_names, policy_names)

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
        return RouterQueryEngine.from_defaults(query_engine_tools=[summary_tool, detail_tool])

    def _build_cached_query_components(self):
        self.postprocessors_default = self._build_postprocessors(candidate_pool_size=FUSION_TOP_K)
        self.bm25_retriever_default = BM25Retriever.from_defaults(nodes=self.standard_nodes, similarity_top_k=BM25_TOP_K)
        self.detail_retriever_default = self._build_detail_retriever()
        self.detail_engine_default = self._build_detail_engine()
        self.summary_engine_full = self._build_summary_engine()
        self.router_engine_default = self._build_router_engine()

    def load(self):
        missing = []
        for required_path in [MANIFEST_PATH, KB_STATS_PATH, CHROMA_DIR]:
            if not os.path.exists(required_path):
                missing.append(required_path)
        if missing:
            raise FileNotFoundError(
                "Required KB artifacts are missing. Run `python prepare_kb.py` first.\n"
                + "\n".join(missing)
            )

        log("loading vector database")
        self.documents = self._load_documents_from_manifest()
        if not self.documents:
            raise RuntimeError("Manifest loaded, but no documents were reconstructed from cached page markdown.")

        self.build_nodes()
        self.configure_models()
        self._reset_query_cache()

        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=Settings.embed_model)
        self._build_cached_query_components()
        self.is_ready = True

        log(f"KB load success: collection count={chroma_collection.count()}")
        log("loading/rebuilding retrievers/query engines completed")

    def _classify_question_rule_based(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip().lower()
        summary_markers = {
            "strong": ["summarize", "summary", "overview", "executive summary", "high level", "broad overview", "main themes", "key themes", "document-wide"],
            "weak": ["broad", "core", "key", "main", "themes", "framework", "governance", "challenges", "what are", "list", "describe", "explain"],
        }
        detail_markers = {
            "strong": ["page", "quote", "exact", "exactly", "what is", "define", "clause", "threshold", "table", "section", "cite", "citation", "according to", "what page"],
            "weak": ["number", "who is responsible", "when", "where", "requirement", "control", "evidence", "definition"],
        }

        summary_strong = [marker for marker in summary_markers["strong"] if text_matches_marker(q, marker)]
        summary_weak = [marker for marker in summary_markers["weak"] if text_matches_marker(q, marker)]
        detail_strong = [marker for marker in detail_markers["strong"] if text_matches_marker(q, marker)]
        detail_weak = [marker for marker in detail_markers["weak"] if text_matches_marker(q, marker)]

        summary_score = len(summary_strong) * 2 + len(summary_weak)
        detail_score = len(detail_strong) * 2 + len(detail_weak)

        if detail_score == 0 and summary_score == 0:
            return {"mode": None, "strong_match": False, "reason": "No strong rule markers found."}
        if detail_score > summary_score:
            return {
                "mode": "detail",
                "strong_match": len(detail_strong) > 0 or detail_score - summary_score >= 2,
                "reason": f"Detail markers matched: {', '.join(detail_strong + detail_weak[:2])}.",
            }
        if summary_score > detail_score:
            return {
                "mode": "summary",
                "strong_match": len(summary_strong) > 0 or summary_score - detail_score >= 2,
                "reason": f"Summary markers matched: {', '.join(summary_strong + summary_weak[:2])}.",
            }
        return {"mode": None, "strong_match": False, "reason": "Rule signals were mixed, so routing was deferred to the LLM router."}

    def _extract_citations(self, source_nodes) -> List[Dict[str, Any]]:
        citations = []
        for index, source_node in enumerate(source_nodes or [], start=1):
            node = source_node.node
            meta = node.metadata or {}
            citation_label = meta.get("citation_label")
            if not citation_label:
                if meta.get("page_number") is not None:
                    citation_label = f"{meta.get('source_file')} p.{meta.get('page_number')}"
                else:
                    citation_label = str(meta.get("source_file"))

            preview = meta.get("window") or meta.get("original_text") or node.text
            citations.append({
                "rank": index,
                "score": getattr(source_node, "score", None),
                "source_file": meta.get("source_file"),
                "page_number": meta.get("page_number"),
                "doc_id": meta.get("doc_id"),
                "citation_label": citation_label,
                "document_type": safe_meta_value(meta.get("document_type")),
                "policy_name": safe_meta_value(meta.get("policy_name")),
                "model_name": safe_meta_value(meta.get("model_name")),
                "version": safe_meta_value(meta.get("version")),
                "section_title": safe_meta_value(meta.get("section_title")),
                "section_path": safe_meta_value(meta.get("section_path")),
                "retrieval_style": meta.get("retrieval_style"),
                "text_preview": short_preview(preview, 700),
            })
        return citations

    def _prepare_citations(self, citations: List[Dict[str, Any]], limit: int = MAX_DISPLAY_CITATIONS) -> List[Dict[str, Any]]:
        deduped = []
        seen = set()
        for citation in citations:
            key = (
                citation.get("source_file"),
                citation.get("page_number"),
                citation.get("doc_id"),
                citation.get("citation_label"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(citation)
            if len(deduped) >= limit:
                break
        for index, citation in enumerate(deduped, start=1):
            citation["rank"] = index
        return deduped

    def _retrieve_supporting_citations(
        self,
        question: str,
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        document_types: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        policy_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        retriever = self._build_detail_retriever(source_files, doc_ids, page_numbers, document_types, model_names, policy_names)
        source_nodes = retriever.retrieve(question)
        citations = self._extract_citations(source_nodes)
        return self._prepare_citations(citations)

    def _is_empty_answer(self, answer_text: str) -> bool:
        normalized = (answer_text or "").strip()
        if not normalized:
            return True
        return normalized.lower() in {"empty response", "none", "null", "no response"}

    def _infer_mode_from_response(self, response: Any) -> str:
        source_nodes = getattr(response, "source_nodes", None) or []
        return "detail" if source_nodes else "summary"

    def _score_confidence(self, citations: List[Dict[str, Any]], mode_used: str) -> Dict[str, Any]:
        citation_count = len(citations)
        page_grounded = sum(1 for citation in citations if citation.get("page_number") is not None)
        source_file_count = len({citation.get("source_file") for citation in citations if citation.get("source_file")})
        scores = [float(citation["score"]) for citation in citations if isinstance(citation.get("score"), (float, int))]
        avg_score = (sum(scores) / len(scores)) if scores else None
        max_score = max(scores) if scores else None

        score_points = 0
        if citation_count >= 4:
            score_points += 3
        elif citation_count >= 2:
            score_points += 2
        elif citation_count == 1:
            score_points += 1

        if page_grounded >= 3:
            score_points += 3
        elif page_grounded >= 1:
            score_points += 2

        if max_score is not None:
            if avg_score is not None and avg_score >= 0.70:
                score_points += 3
            elif avg_score is not None and avg_score >= 0.40:
                score_points += 2
            else:
                score_points += 1
        elif mode_used == "summary" and citation_count >= 2:
            score_points += 1

        if score_points >= 7:
            label = "High"
        elif score_points >= 4:
            label = "Medium"
        else:
            label = "Low"

        reason_parts = [f"{citation_count} citation(s)", f"{page_grounded} page-grounded", f"{source_file_count} file(s)"]
        if avg_score is not None:
            reason_parts.append(f"avg retrieval score {avg_score:.3f}")
        else:
            reason_parts.append("no retrieval score metadata")

        return {
            "label": label,
            "reason": ", ".join(reason_parts),
            "citation_count": citation_count,
            "page_grounded_count": page_grounded,
            "source_file_count": source_file_count,
        }

    def _is_weak_result(self, answer_text: str, citations: List[Dict[str, Any]], confidence: Dict[str, Any]) -> bool:
        if self._is_empty_answer(answer_text):
            return True
        if confidence["label"] == "Low":
            return True
        if not citations:
            return True
        return False

    def _build_engine_for_mode(
        self,
        mode: str,
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        document_types: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        policy_names: Optional[List[str]] = None,
    ):
        if mode == "summary":
            return self._build_summary_engine(source_files, doc_ids, page_numbers, document_types, model_names, policy_names)
        if mode == "detail":
            return self._build_detail_engine(source_files, doc_ids, page_numbers, document_types, model_names, policy_names)
        return self._build_router_engine(source_files, doc_ids, page_numbers, document_types, model_names, policy_names)

    def _run_mode_query(
        self,
        mode: str,
        question: str,
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        document_types: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        policy_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        engine = self._build_engine_for_mode(mode, source_files, doc_ids, page_numbers, document_types, model_names, policy_names)
        response = engine.query(question)
        answer_text = str(response).strip()
        if mode == "summary":
            citations = self._retrieve_supporting_citations(question, source_files, doc_ids, page_numbers, document_types, model_names, policy_names)
        else:
            citations = self._prepare_citations(self._extract_citations(getattr(response, "source_nodes", []) or []))
        return {"mode_used": mode, "response": response, "answer_text": answer_text, "citations": citations}

    def _run_router_query(
        self,
        question: str,
        source_files: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        document_types: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        policy_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        engine = self._build_router_engine(source_files, doc_ids, page_numbers, document_types, model_names, policy_names)
        response = engine.query(question)
        mode_used = self._infer_mode_from_response(response)
        answer_text = str(response).strip()
        if mode_used == "summary":
            citations = self._retrieve_supporting_citations(question, source_files, doc_ids, page_numbers, document_types, model_names, policy_names)
        else:
            citations = self._prepare_citations(self._extract_citations(getattr(response, "source_nodes", []) or []))
        return {"mode_used": mode_used, "response": response, "answer_text": answer_text, "citations": citations}

    def _build_conservative_answer(self, question: str, confidence_reason: str) -> str:
        return (
            "I could not verify a strong grounded answer from the current retrieval results. "
            f"Please treat this as uncertain. Confidence basis: {confidence_reason}\n\n"
            f"Question: {question}"
        )

    def answer(self, question: str, mode: str = "router") -> Dict[str, Any]:
        if not self.is_ready or self.vector_index is None:
            return {
                "answer": "The knowledge base is empty or not loaded.",
                "citations": [],
                "mode_used": mode,
                "routing_method": "none",
                "confidence": {"label": "Low", "reason": "KB not ready."},
            }

        log(f"mode selected: {mode}")
        log(f"question received: {question}")
        log("retrieval started")

        last_error = None
        final_answer = ""
        citations: List[Dict[str, Any]] = []
        mode_used = mode
        routing_method = "direct"
        fallback_steps: List[str] = []
        confidence = {"label": "Low", "reason": "No evidence available."}

        try:
            if mode == "router":
                rule_route = self._classify_question_rule_based(question)
                try:
                    if rule_route["strong_match"] and rule_route["mode"] in {"detail", "summary"}:
                        routing_method = f"rule-based ({rule_route['reason']})"
                        result = self._run_mode_query(rule_route["mode"], question)
                    else:
                        routing_method = f"llm-router ({rule_route['reason']})"
                        result = self._run_router_query(question)
                    final_answer = result["answer_text"]
                    citations = result["citations"]
                    mode_used = result["mode_used"]
                    confidence = self._score_confidence(citations, mode_used)
                except Exception as error:
                    last_error = error
                    final_answer = ""
                    citations = []
                    mode_used = "router"
                    confidence = self._score_confidence(citations, mode_used)

                if self._is_weak_result(final_answer, citations, confidence):
                    if mode_used != "detail" or self._is_empty_answer(final_answer):
                        fallback_steps.append("detail")
                        try:
                            result = self._run_mode_query("detail", question)
                            final_answer = result["answer_text"]
                            citations = result["citations"]
                            mode_used = result["mode_used"]
                            confidence = self._score_confidence(citations, mode_used)
                        except Exception as error:
                            last_error = error

                if self._is_weak_result(final_answer, citations, confidence):
                    if mode_used != "summary" or self._is_empty_answer(final_answer):
                        fallback_steps.append("summary")
                        try:
                            result = self._run_mode_query("summary", question)
                            final_answer = result["answer_text"]
                            citations = result["citations"]
                            mode_used = result["mode_used"]
                            confidence = self._score_confidence(citations, mode_used)
                        except Exception as error:
                            last_error = error

                if self._is_weak_result(final_answer, citations, confidence):
                    fallback_steps.append("conservative")
                    final_answer = self._build_conservative_answer(question, confidence["reason"])
            else:
                result = self._run_mode_query(mode, question)
                final_answer = result["answer_text"]
                citations = result["citations"]
                mode_used = result["mode_used"]
                confidence = self._score_confidence(citations, mode_used)
        except Exception as error:
            last_error = error

        if not final_answer:
            if citations:
                final_answer = "I found relevant source passages, but the model returned an empty answer. Review the cited passages below."
            elif last_error is not None:
                final_answer = f"Query failed: {last_error}"
            else:
                final_answer = "The model returned an empty answer."
            confidence = self._score_confidence(citations, mode_used)

        log("answer generated")
        log(f"number of source nodes returned: {len(citations)}")
        return {
            "answer": final_answer,
            "citations": citations,
            "mode_used": mode_used,
            "routing_method": routing_method,
            "fallback_steps": fallback_steps,
            "confidence": confidence,
        }


def print_banner(stats: Dict[str, Any]):
    print("=" * 78)
    print("RAG Terminal Chat")
    print("=" * 78)
    print("Commands: /mode router|summary|detail, /clear, /help, exit, quit, q")
    print(f"KB loaded message: {Path(WORK_DIR).resolve()}")
    print(f"number of indexed records: {stats.get('sentence_window_node_count', 'unknown')}")
    print("available modes: router, summary, detail")
    print("=" * 78)


def print_help():
    print("Enter a question to query the KB.")
    print("Use `/mode router`, `/mode summary`, or `/mode detail` to switch mode.")
    print("Use `/clear` to clear conversation history.")
    print("Type `exit`, `quit`, or `q` to stop.")


def main():
    kb = TerminalRoutedKnowledgeBase()
    try:
        kb.load()
    except Exception as error:
        print(f"KB load failed: {error}")
        sys.exit(1)

    print_banner(kb.stats)

    current_mode = "router"
    history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input(f"\n[{current_mode}] Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"exit", "quit", "q"}:
            print("Exiting.")
            break
        if lowered == "/help":
            print_help()
            continue
        if lowered == "/clear":
            history.clear()
            print("Conversation history cleared.")
            continue
        if lowered.startswith("/mode "):
            next_mode = lowered.split(maxsplit=1)[1].strip()
            if next_mode not in {"router", "summary", "detail"}:
                print(f"Unsupported mode: {next_mode}")
                continue
            current_mode = next_mode
            print(f"Mode switched to: {current_mode}")
            continue

        question_for_query = build_conversational_question(user_input, history)
        result = kb.answer(question_for_query, mode=current_mode)

        print("\nAnswer:")
        print(result["answer"])
        print()
        print(f"Mode used: {result['mode_used']}")
        print(f"Routing method: {result['routing_method']}")
        print(
            "Evidence: "
            f"{result['confidence'].get('citation_count', 0)} citation(s), "
            f"{result['confidence'].get('page_grounded_count', 0)} page-grounded, "
            f"{result['confidence'].get('source_file_count', 0)} file(s)"
        )
        print(f"Confidence: {result['confidence'].get('label')} ({result['confidence'].get('reason')})")
        print()
        print("Sources:")
        print(format_sources_text(result["citations"]))

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()
