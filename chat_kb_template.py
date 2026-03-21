import os
import re
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import chromadb
import litellm

from llama_index.core import Document, Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.llms.litellm import LiteLLM
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

try:
    from utils.cortex_litellm_provider import register_cortex_provider
except Exception:
    def register_cortex_provider():
        return None

PROFILE_CONFIGS = {
    "policy_guidance": {
        "description": "Policy, risk triage, and validation guidance KB",
        "collection_name": "policy_guidance_collection",
        "example_questions": [
            "What guidance is relevant for validation scope?",
            "What are the main governance expectations?",
            "Give risk triage guidance for a high-materiality AI use case.",
        ],
    },
    "model_evidence": {
        "description": "Model documentation and model evidence KB",
        "collection_name": "model_evidence_collection",
        "example_questions": [
            "What does this model do?",
            "What assumptions are described?",
            "Check whether the documentation appears complete.",
        ],
    },
}

CORTEX_LLM_MODEL = os.getenv("CORTEX_LLM_MODEL", "cortex/vertex_ai/gemini-2.5-flash")
CORTEX_EMBED_MODEL = os.getenv("CORTEX_EMBED_MODEL", "cortex/vertex_ai/text-embedding-004")
CORTEX_CUSTOM_LLM_PROVIDER = os.getenv("CORTEX_CUSTOM_LLM_PROVIDER", "cortex")
CORTEX_LLM_TEMPERATURE = float(os.getenv("CORTEX_LLM_TEMPERATURE", "0.1"))
CORTEX_LLM_MAX_TOKENS = int(os.getenv("CORTEX_LLM_MAX_TOKENS", "1024"))
CORTEX_CONTEXT_WINDOW = int(os.getenv("CORTEX_CONTEXT_WINDOW", "32768"))

BASE_CHUNK_SIZE = 900
BASE_CHUNK_OVERLAP = 120
WINDOW_SIZE = 3
VECTOR_TOP_K = 6
BM25_TOP_K = 6
FUSION_TOP_K = 6
MAX_DISPLAY_CITATIONS = 5
USE_RERANKER = False
RERANK_TOP_N = 3
RERANK_MIN_FUSION_CANDIDATES = 8
MAX_SECTION_DOC_CHARS = 3200
DETAIL_RESPONSE_MODE = "compact"
SUMMARY_RESPONSE_MODE = "tree_summarize"
DEBUG_RETRIEVAL = True


def log(message: str):
    print(f"[chat_kb] {message}", flush=True)


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
    pattern = r"\b" + re.escape(marker.lower()) + r"\b"
    return re.search(pattern, (text or "").lower()) is not None


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
    return [{"section_title": "Document Overview", "section_path": "Document Overview", "text": normalized[:MAX_SECTION_DOC_CHARS], "level": 1}]


def build_metadata_filters(**kwargs) -> Optional[MetadataFilters]:
    key_map = {
        "source_files": "source_file",
        "doc_ids": "doc_id",
        "page_numbers": "page_number",
        "document_types": "document_type",
        "model_names": "model_name",
        "policy_names": "policy_name",
        "section_types": "section_type",
        "risk_domains": "risk_domain",
        "validation_topics": "validation_topic",
    }
    filters = []
    for arg_key, meta_key in key_map.items():
        values = kwargs.get(arg_key)
        if values:
            for value in values:
                filters.append(MetadataFilter(key=meta_key, value=value, operator=FilterOperator.EQ))
    if not filters:
        return None
    return MetadataFilters(filters=filters)


def filter_documents(documents: List[Document], **kwargs) -> List[Document]:
    filtered = []
    for document in documents:
        meta = document.metadata or {}
        ok = True
        for arg_key, meta_key in {
            "source_files": "source_file",
            "doc_ids": "doc_id",
            "page_numbers": "page_number",
            "document_types": "document_type",
            "model_names": "model_name",
            "policy_names": "policy_name",
            "section_types": "section_type",
            "risk_domains": "risk_domain",
            "validation_topics": "validation_topic",
        }.items():
            values = kwargs.get(arg_key)
            if values and meta.get(meta_key) not in values and safe_meta_value(meta.get(meta_key)) not in values:
                ok = False
                break
        if ok:
            filtered.append(document)
    return filtered


def filter_nodes(nodes, **kwargs):
    filtered = []
    for node in nodes:
        meta = node.metadata or {}
        ok = True
        for arg_key, meta_key in {
            "source_files": "source_file",
            "doc_ids": "doc_id",
            "page_numbers": "page_number",
            "document_types": "document_type",
            "model_names": "model_name",
            "policy_names": "policy_name",
            "section_types": "section_type",
            "risk_domains": "risk_domain",
            "validation_topics": "validation_topic",
        }.items():
            values = kwargs.get(arg_key)
            if values and meta.get(meta_key) not in values and safe_meta_value(meta.get(meta_key)) not in values:
                ok = False
                break
        if ok:
            filtered.append(node)
    return filtered


def format_sources_text(citations: List[Dict[str, Any]]) -> str:
    if not citations:
        return "No page-level source nodes returned."
    lines = []
    for citation in citations:
        lines.append(
            f"[{citation['rank']}] {citation['citation_label']} | section={citation.get('section_title')} | "
            f"retrieval={citation.get('retrieval_style')} | score={citation.get('score')}"
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


class CortexLiteLLMEmbedding(BaseEmbedding):
    model_name: str = CORTEX_EMBED_MODEL
    custom_llm_provider: str = CORTEX_CUSTOM_LLM_PROVIDER

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = litellm.embedding(model=self.model_name, custom_llm_provider=self.custom_llm_provider, input=texts)
        if not response or "data" not in response:
            raise RuntimeError(f"Unexpected embedding response: {response}")
        vectors = []
        for item in response["data"]:
            embedding = item.get("embedding")
            if embedding is None:
                raise RuntimeError(f"Missing embedding in response item: {item}")
            vectors.append(embedding)
        return vectors

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._embed_texts([query])
        return embeddings[0] if embeddings else []

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._embed_texts([text])
        return embeddings[0] if embeddings else []

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts)


class TerminalRoutedKnowledgeBase:
    def __init__(self, profile: str, work_root: str):
        if profile not in PROFILE_CONFIGS:
            raise ValueError(f"Unsupported profile: {profile}")
        self.profile = profile
        self.work_root = work_root
        self.collection_name = PROFILE_CONFIGS[profile]["collection_name"]
        self.base_dir = os.path.join(work_root, profile)
        self.chroma_dir = os.path.join(self.base_dir, "chroma_db")
        self.manifest_path = os.path.join(self.base_dir, "manifest.json")
        self.stats_path = os.path.join(self.base_dir, "kb_stats.json")

        raw_manifest = load_json(self.manifest_path, {})
        if isinstance(raw_manifest, dict) and isinstance(raw_manifest.get("documents"), dict):
            self.manifest = raw_manifest.get("documents", {})
            self.manifest_meta = raw_manifest.get("meta", {})
        else:
            self.manifest = raw_manifest if isinstance(raw_manifest, dict) else {}
            self.manifest_meta = {}
        self.stats = load_json(self.stats_path, {})

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
        register_cortex_provider()
        Settings.llm = LiteLLM(
            model=CORTEX_LLM_MODEL,
            temperature=CORTEX_LLM_TEMPERATURE,
            max_tokens=CORTEX_LLM_MAX_TOKENS,
            context_window=CORTEX_CONTEXT_WINDOW,
        )
        Settings.embed_model = CortexLiteLLMEmbedding(model_name=CORTEX_EMBED_MODEL, custom_llm_provider=CORTEX_CUSTOM_LLM_PROVIDER)

    def _log_runtime_configuration(self):
        log(f"KB profile: {self.profile}")
        log(f"Cortex LLM model: {CORTEX_LLM_MODEL}")
        log(f"Cortex embedding model: {CORTEX_EMBED_MODEL}")
        log(f"Cortex provider registration: {CORTEX_CUSTOM_LLM_PROVIDER}")
        log(f"Cortex context window: {CORTEX_CONTEXT_WINDOW}")
        stored_embed_model = self.stats.get("embedding_model")
        stored_provider = self.stats.get("embedding_provider")
        if stored_embed_model:
            log(f"Prepared KB embedding model: {stored_embed_model}")
        if stored_provider:
            log(f"Prepared KB embedding provider: {stored_provider}")
        if stored_embed_model and stored_embed_model != CORTEX_EMBED_MODEL:
            raise RuntimeError(f"Embedding model mismatch. KB was built with {stored_embed_model}, but chat is using {CORTEX_EMBED_MODEL}. Re-run prepare_kb_template.py.")

    def _reset_query_cache(self):
        self.detail_engine_default = None
        self.detail_retriever_default = None
        self.summary_engine_full = None
        self.router_engine_default = None
        self.bm25_retriever_default = None
        self.postprocessors_default = None

    def _has_active_filters(self, **kwargs) -> bool:
        return any(bool(v) for v in kwargs.values())

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
        postprocessors = [MetadataReplacementPostProcessor(target_metadata_key="window")]
        if USE_RERANKER and candidate_pool_size >= RERANK_MIN_FUSION_CANDIDATES:
            postprocessors.append(LLMRerank(top_n=RERANK_TOP_N, llm=Settings.llm))
        return postprocessors

    def _build_detail_retriever_components(self, **filters):
        metadata_filters = build_metadata_filters(**filters)
        vector_retriever = VectorIndexRetriever(index=self.vector_index, similarity_top_k=VECTOR_TOP_K, filters=metadata_filters)
        if not self._has_active_filters(**filters) and self.bm25_retriever_default is not None:
            bm25_retriever = self.bm25_retriever_default
        else:
            bm25_nodes = filter_nodes(self.standard_nodes, **filters)
            bm25_retriever = BM25Retriever.from_defaults(nodes=bm25_nodes, similarity_top_k=BM25_TOP_K)
        fusion_retriever = QueryFusionRetriever([vector_retriever, bm25_retriever], similarity_top_k=FUSION_TOP_K, num_queries=1, mode="reciprocal_rerank", use_async=False)
        return vector_retriever, bm25_retriever, fusion_retriever

    def _build_detail_retriever(self, **filters):
        if not self._has_active_filters(**filters) and self.detail_retriever_default is not None:
            return self.detail_retriever_default
        _, _, fusion_retriever = self._build_detail_retriever_components(**filters)
        return fusion_retriever

    def _build_detail_engine(self, **filters):
        if not self._has_active_filters(**filters) and self.detail_engine_default is not None:
            return self.detail_engine_default
        fusion_retriever = self._build_detail_retriever(**filters)
        return RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            response_mode=DETAIL_RESPONSE_MODE,
            node_postprocessors=self.postprocessors_default if not self._has_active_filters(**filters) and self.postprocessors_default is not None else self._build_postprocessors(candidate_pool_size=FUSION_TOP_K),
        )

    def _build_summary_engine(self, **filters):
        if not self._has_active_filters(**filters) and self.summary_engine_full is not None:
            return self.summary_engine_full
        filtered_docs = filter_documents(self.summary_documents or self.documents, **filters)
        if not filtered_docs:
            filtered_docs = self.summary_documents or self.documents
        summary_index = SummaryIndex.from_documents(filtered_docs)
        return summary_index.as_query_engine(response_mode=SUMMARY_RESPONSE_MODE)

    def _build_router_engine(self, **filters):
        if not self._has_active_filters(**filters) and self.router_engine_default is not None:
            return self.router_engine_default
        detail_engine = self._build_detail_engine(**filters)
        summary_engine = self._build_summary_engine(**filters)
        summary_tool = QueryEngineTool.from_defaults(query_engine=summary_engine, description="Use for overview, synthesis, and broad understanding.")
        detail_tool = QueryEngineTool.from_defaults(query_engine=detail_engine, description="Use for evidence, citations, definitions, specifics, and exact support.")
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
        for required_path in [self.manifest_path, self.stats_path, self.chroma_dir]:
            if not os.path.exists(required_path):
                missing.append(required_path)
        if missing:
            raise FileNotFoundError("Required KB artifacts are missing. Run prepare_kb_template.py first.\n" + "\n".join(missing))

        log("loading vector database")
        self.documents = self._load_documents_from_manifest()
        if not self.documents:
            raise RuntimeError("Manifest loaded, but no documents were reconstructed from cached page markdown.")
        self.build_nodes()
        self.configure_models()
        self._log_runtime_configuration()
        self._reset_query_cache()
        chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        chroma_collection = chroma_client.get_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=Settings.embed_model)
        self._build_cached_query_components()
        self.is_ready = True
        log(f"KB load success: collection count={chroma_collection.count()}")
        log("loading/rebuilding retrievers/query engines completed")

    def _classify_question_rule_based(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip().lower()
        summary_markers = {
            "strong": ["summarize", "summary", "overview", "executive summary", "high level", "broad overview", "main themes", "key themes"],
            "weak": ["broad", "core", "key", "main", "themes", "framework", "governance", "what are", "list", "describe", "explain"],
        }
        detail_markers = {
            "strong": ["page", "quote", "exact", "define", "clause", "threshold", "table", "section", "cite", "citation", "according to", "what page", "where does it say"],
            "weak": ["what is", "number", "requirement", "control", "evidence", "definition"],
        }
        if self.profile == "policy_guidance":
            summary_markers["weak"].extend(["validation scope", "triage", "guidance"])
            detail_markers["weak"].extend(["control", "obligation", "role"])
        else:
            summary_markers["weak"].extend(["model overview", "completeness"])
            detail_markers["weak"].extend(["input", "output", "assumption", "limitation"])

        summary_strong = [m for m in summary_markers["strong"] if text_matches_marker(q, m)]
        summary_weak = [m for m in summary_markers["weak"] if text_matches_marker(q, m)]
        detail_strong = [m for m in detail_markers["strong"] if text_matches_marker(q, m)]
        detail_weak = [m for m in detail_markers["weak"] if text_matches_marker(q, m)]
        summary_score = len(summary_strong) * 2 + len(summary_weak)
        detail_score = len(detail_strong) * 2 + len(detail_weak)
        if detail_score == 0 and summary_score == 0:
            return {"mode": None, "strong_match": False, "reason": "No strong rule markers found."}
        if detail_score > summary_score and (len(detail_strong) > 0 or detail_score - summary_score >= 2):
            return {"mode": "detail", "strong_match": True, "reason": f"Detail markers matched: {', '.join(detail_strong + detail_weak[:2])}."}
        if summary_score > detail_score and (len(summary_strong) > 0 or summary_score - detail_score >= 2):
            return {"mode": "summary", "strong_match": True, "reason": f"Summary markers matched: {', '.join(summary_strong + summary_weak[:2])}."}
        return {"mode": None, "strong_match": False, "reason": "Rule signals were weak or mixed, so routing was deferred to the LLM router."}

    def _extract_citations(self, source_nodes) -> List[Dict[str, Any]]:
        citations = []
        for index, source_node in enumerate(source_nodes or [], start=1):
            node = source_node.node
            meta = node.metadata or {}
            citation_label = meta.get("citation_label") or (f"{meta.get('source_file')} p.{meta.get('page_number')}" if meta.get("page_number") is not None else str(meta.get("source_file")))
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
                "section_type": safe_meta_value(meta.get("section_type")),
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
            key = (citation.get("source_file"), citation.get("page_number"), citation.get("doc_id"), citation.get("citation_label"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(citation)
            if len(deduped) >= limit:
                break
        for index, citation in enumerate(deduped, start=1):
            citation["rank"] = index
        return deduped

    def _retrieve_supporting_citations(self, question: str, **filters) -> List[Dict[str, Any]]:
        retriever = self._build_detail_retriever(**filters)
        source_nodes = retriever.retrieve(question)
        citations = self._extract_citations(source_nodes)
        return self._prepare_citations(citations)

    def _is_empty_answer(self, answer_text: str) -> bool:
        normalized = (answer_text or "").strip()
        return (not normalized) or normalized.lower() in {"empty response", "none", "null", "no response"}

    def _infer_mode_from_response(self, response: Any) -> str:
        source_nodes = getattr(response, "source_nodes", None) or []
        return "detail" if source_nodes else "summary"

    def _score_confidence(self, citations: List[Dict[str, Any]], mode_used: str) -> Dict[str, Any]:
        citation_count = len(citations)
        page_grounded = sum(1 for citation in citations if citation.get("page_number") is not None)
        source_file_count = len({citation.get("source_file") for citation in citations if citation.get("source_file")})
        score_points = 0
        if citation_count >= 4: score_points += 3
        elif citation_count >= 2: score_points += 2
        elif citation_count == 1: score_points += 1
        if page_grounded >= 3: score_points += 3
        elif page_grounded >= 1: score_points += 2
        if mode_used == "summary" and citation_count >= 2: score_points += 1
        label = "High" if score_points >= 6 else "Medium" if score_points >= 3 else "Low"
        reason_parts = [f"{citation_count} citation(s)", f"{page_grounded} page-grounded", f"{source_file_count} file(s)"]
        if citation_count >= 3 and page_grounded >= 2: reason_parts.append("multiple grounded sources available")
        elif citation_count >= 1: reason_parts.append("limited but usable supporting evidence")
        else: reason_parts.append("no grounded supporting evidence")
        return {"label": label, "reason": ", ".join(reason_parts), "citation_count": citation_count, "page_grounded_count": page_grounded, "source_file_count": source_file_count}

    def _is_weak_result(self, answer_text: str, citations: List[Dict[str, Any]], confidence: Dict[str, Any]) -> bool:
        return self._is_empty_answer(answer_text) or confidence["label"] == "Low" or not citations

    def _build_engine_for_mode(self, mode: str, **filters):
        if mode == "summary":
            return self._build_summary_engine(**filters)
        if mode == "detail":
            return self._build_detail_engine(**filters)
        return self._build_router_engine(**filters)

    def _debug_retrieval_pipeline(self, question: str, **filters):
        if not DEBUG_RETRIEVAL:
            return
        try:
            vector_retriever, bm25_retriever, fusion_retriever = self._build_detail_retriever_components(**filters)
            vector_hits = vector_retriever.retrieve(question)
            bm25_hits = bm25_retriever.retrieve(question)
            fusion_hits = fusion_retriever.retrieve(question)
            log(f"debug vector hits: {len(vector_hits)}")
            for hit in vector_hits[:2]:
                meta = hit.node.metadata or {}
                log(f"  vector -> {meta.get('citation_label')} | score={getattr(hit, 'score', None)}")
            log(f"debug bm25 hits: {len(bm25_hits)}")
            for hit in bm25_hits[:2]:
                meta = hit.node.metadata or {}
                log(f"  bm25 -> {meta.get('citation_label')} | score={getattr(hit, 'score', None)}")
            log(f"debug fusion hits: {len(fusion_hits)}")
            for hit in fusion_hits[:3]:
                meta = hit.node.metadata or {}
                log(f"  fusion -> {meta.get('citation_label')} | score={getattr(hit, 'score', None)}")
        except Exception as error:
            log(f"debug retrieval pipeline failed: {error}")

    def _task_prompt(self, task: str, question: str) -> str:
        if task == "scope_guidance" and self.profile == "policy_guidance":
            return (
                "Using the retrieved policy and validation guidance, provide: \n"
                "1. suggested validation scope areas\n2. relevant risk themes\n3. evidence/documents expected\n4. caveats and dependencies\n\n"
                f"Question: {question}"
            )
        if task == "triage_guidance" and self.profile == "policy_guidance":
            return (
                "Using the retrieved triage and governance material, provide:\n"
                "1. likely triage considerations\n2. materiality/risk indicators\n3. governance checks\n4. recommended next validation steps\n\n"
                f"Question: {question}"
            )
        if task == "validation_steps" and self.profile == "policy_guidance":
            return (
                "Using the retrieved policy/guidance corpus, give a practical step-by-step validation approach with citations.\n\n"
                f"Question: {question}"
            )
        if task == "completeness_check" and self.profile == "model_evidence":
            return (
                "Review the retrieved model documentation and provide:\n"
                "1. what information appears present\n2. what appears missing or unclear\n3. follow-up questions\n4. supporting citations\n\n"
                f"Question: {question}"
            )
        return question

    def _default_filters_for_task(self, task: str) -> Dict[str, Any]:
        if self.profile == "policy_guidance":
            if task == "scope_guidance":
                return {"validation_topics": ["validation_scope"]}
            if task == "triage_guidance":
                return {"validation_topics": ["risk_triage"]}
            if task == "validation_steps":
                return {"document_types": ["policy", "guidance", "methodology", "standard"]}
        else:
            if task == "completeness_check":
                return {"document_types": ["model_document", "implementation_note", "validation_report", "unknown"]}
        return {}

    def _run_mode_query(self, mode: str, question: str, **filters) -> Dict[str, Any]:
        if mode == "detail":
            self._debug_retrieval_pipeline(question, **filters)
        engine = self._build_engine_for_mode(mode, **filters)
        response = engine.query(question)
        answer_text = str(response).strip()
        if mode == "summary":
            citations = self._retrieve_supporting_citations(question, **filters)
        else:
            raw_source_nodes = getattr(response, "source_nodes", []) or []
            log(f"debug final source_nodes after query engine: {len(raw_source_nodes)}")
            citations = self._prepare_citations(self._extract_citations(raw_source_nodes))
        return {"mode_used": mode, "response": response, "answer_text": answer_text, "citations": citations}

    def _run_router_query(self, question: str, **filters) -> Dict[str, Any]:
        engine = self._build_router_engine(**filters)
        response = engine.query(question)
        mode_used = self._infer_mode_from_response(response)
        answer_text = str(response).strip()
        if mode_used == "summary":
            citations = self._retrieve_supporting_citations(question, **filters)
        else:
            raw_source_nodes = getattr(response, "source_nodes", []) or []
            log(f"debug final source_nodes after router engine: {len(raw_source_nodes)}")
            citations = self._prepare_citations(self._extract_citations(raw_source_nodes))
        return {"mode_used": mode_used, "response": response, "answer_text": answer_text, "citations": citations}

    def _build_conservative_answer(self, question: str, confidence_reason: str) -> str:
        return (
            "I could not verify a strong grounded answer from the current retrieval results. "
            f"Please treat this as uncertain. Confidence basis: {confidence_reason}\n\nQuestion: {question}"
        )

    def answer(self, question: str, mode: str = "router", task: str = "qa") -> Dict[str, Any]:
        if not self.is_ready or self.vector_index is None:
            return {"answer": "The knowledge base is empty or not loaded.", "citations": [], "mode_used": mode, "routing_method": "none", "initial_route": mode, "confidence": {"label": "Low", "reason": "KB not ready."}, "fallback_steps": []}

        enhanced_question = self._task_prompt(task, question)
        filters = self._default_filters_for_task(task)

        log(f"mode selected: {mode}")
        log(f"task selected: {task}")
        log(f"question received: {question}")
        log("retrieval started")

        last_error = None
        final_answer = ""
        citations: List[Dict[str, Any]] = []
        mode_used = mode
        routing_method = "direct"
        initial_route = mode
        fallback_steps: List[str] = []
        confidence = {"label": "Low", "reason": "No evidence available."}

        try:
            if mode == "router":
                rule_route = self._classify_question_rule_based(enhanced_question)
                try:
                    if rule_route["strong_match"] and rule_route["mode"] in {"detail", "summary"}:
                        initial_route = rule_route["mode"]
                        routing_method = f"rule-based ({rule_route['reason']})"
                        result = self._run_mode_query(rule_route["mode"], enhanced_question, **filters)
                    else:
                        initial_route = "llm-router"
                        routing_method = f"llm-router ({rule_route['reason']})"
                        result = self._run_router_query(enhanced_question, **filters)
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
                            result = self._run_mode_query("detail", enhanced_question, **filters)
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
                            result = self._run_mode_query("summary", enhanced_question, **filters)
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
                initial_route = mode
                result = self._run_mode_query(mode, enhanced_question, **filters)
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
            "initial_route": initial_route,
            "fallback_steps": fallback_steps,
            "confidence": confidence,
        }


def print_banner(profile: str, stats: Dict[str, Any]):
    print("=" * 78)
    print(f"RAG Terminal Chat - {profile}")
    print("=" * 78)
    print("Commands: /mode router|summary|detail, /task qa|summary|scope_guidance|triage_guidance|validation_steps|completeness_check, /clear, /help, exit, quit, q")
    print(f"KB loaded message: {profile}")
    print(f"number of indexed records: {stats.get('sentence_window_node_count', 'unknown')}")
    print("available modes: router, summary, detail")
    print(f"available task examples: {', '.join(PROFILE_CONFIGS[profile]['example_questions'])}")
    print("=" * 78)


def print_help(profile: str):
    print("Enter a question to query the KB.")
    print("Use `/mode router`, `/mode summary`, or `/mode detail` to switch mode.")
    print("Use `/task qa`, `/task summary`, `/task scope_guidance`, `/task triage_guidance`, `/task validation_steps`, or `/task completeness_check`.")
    print("Use `/clear` to clear conversation history.")
    print("Type `exit`, `quit`, or `q` to stop.")
    print("Example questions:")
    for q in PROFILE_CONFIGS[profile]["example_questions"]:
        print(f"- {q}")


def run_examples(kb: TerminalRoutedKnowledgeBase):
    print("=== Example Smoke Tests ===")
    examples = PROFILE_CONFIGS[kb.profile]["example_questions"]
    for q in examples[:2]:
        result = kb.answer(q, mode="router", task="qa")
        print(f"Q: {q}")
        print(f"A: {result['answer'][:300]}{'...' if len(result['answer']) > 300 else ''}")
        print(f"Citations: {len(result['citations'])}")
        print("---")
    print("=== End Smoke Tests ===")


def main():
    parser = argparse.ArgumentParser(description="Chat against a profile-based RAG knowledge base.")
    parser.add_argument("--profile", choices=list(PROFILE_CONFIGS.keys()), required=True)
    parser.add_argument("--work-root", default="./kb_workspaces")
    parser.add_argument("--smoke-test", action="store_true", help="Run built-in example questions then exit.")
    args = parser.parse_args()

    kb = TerminalRoutedKnowledgeBase(profile=args.profile, work_root=args.work_root)
    try:
        kb.load()
    except Exception as error:
        print(f"KB load failed: {error}")
        sys.exit(1)

    if args.smoke_test:
        run_examples(kb)
        return

    print_banner(args.profile, kb.stats)
    current_mode = "router"
    current_task = "qa"
    history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input(f"\n[{args.profile} | {current_mode} | {current_task}] Ask a question: ").strip()
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
            print_help(args.profile)
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
        if lowered.startswith("/task "):
            next_task = lowered.split(maxsplit=1)[1].strip()
            if next_task not in {"qa", "summary", "scope_guidance", "triage_guidance", "validation_steps", "completeness_check"}:
                print(f"Unsupported task: {next_task}")
                continue
            current_task = next_task
            print(f"Task switched to: {current_task}")
            continue

        question_for_query = build_conversational_question(user_input, history)
        result = kb.answer(question_for_query, mode=current_mode, task=current_task)

        print("\nAnswer:")
        print(result["answer"])
        print()
        print(f"Initial route: {result.get('initial_route')}")
        print(f"Mode used: {result['mode_used']}")
        print(f"Routing method: {result['routing_method']}")
        print(f"Fallback steps: {' -> '.join(result['fallback_steps']) if result['fallback_steps'] else 'none'}")
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
