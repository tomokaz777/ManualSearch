import argparse
import bisect
from difflib import SequenceMatcher
import hashlib
import json
import os
import re
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


load_dotenv()


DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma_db"))
MANIFEST_PATH = CHROMA_DIR / "index_manifest.json"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "manual_rag")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local").lower()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
)

USE_OPENAI_CHAT = os.getenv("USE_OPENAI_CHAT", "false").lower() == "true"
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
MAX_GENERATION_TOKENS = int(os.getenv("MAX_GENERATION_TOKENS", "350"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))
RELEVANCE_SCORE_THRESHOLD = float(os.getenv("RELEVANCE_SCORE_THRESHOLD", "0.35"))
MIN_VECTOR_SCORE = float(os.getenv("MIN_VECTOR_SCORE", "0.2"))
MIN_LEXICAL_SCORE = float(os.getenv("MIN_LEXICAL_SCORE", "0.03"))
CANDIDATE_MULTIPLIER = int(os.getenv("CANDIDATE_MULTIPLIER", "6"))

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".yaml", ".yml"}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class FileContent:
    text: str
    page_ranges: List[Tuple[int, int, int]]  # (start_char, end_char, page_number)
    page_texts: Optional[List[str]] = None


class PrefixEmbeddings(Embeddings):
    def __init__(
        self,
        base: Embeddings,
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> None:
        self.base = base
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"{self.doc_prefix}{t}" for t in texts]
        return self.base.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        return self.base.embed_query(f"{self.query_prefix}{text}")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest() -> Dict:
    if not MANIFEST_PATH.exists():
        return {"files": {}}
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: Dict) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


@lru_cache(maxsize=8)
def _build_embeddings(
    embedding_backend: str,
    openai_embedding_model: str,
    local_embedding_model: str,
) -> Embeddings:
    if embedding_backend == "openai":
        return OpenAIEmbeddings(model=openai_embedding_model)
    if embedding_backend == "local":
        # Fallback chain: if the configured local model is broken/corrupted in cache,
        # try smaller multilingual models to keep the app usable.
        candidate_models = [
            local_embedding_model,
            "intfloat/multilingual-e5-base",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]
        errors: List[str] = []
        for model_name in candidate_models:
            try:
                base = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                # E5 models are trained with "query:" / "passage:" prefixes.
                if "e5" in model_name.lower():
                    return PrefixEmbeddings(base=base, query_prefix="query: ", doc_prefix="passage: ")
                return base
            except Exception as e:
                errors.append(f"{model_name}: {e}")
                continue
        raise RuntimeError(
            "Local embeddings initialization failed for all candidate models.\n"
            + "\n".join(errors)
            + "\nPlease check network/cache, or switch EMBEDDING_BACKEND=openai."
        )
    raise ValueError(
        "Invalid EMBEDDING_BACKEND. Use 'openai' or 'local'."
    )


def get_embeddings():
    return _build_embeddings(
        EMBEDDING_BACKEND,
        OPENAI_EMBEDDING_MODEL,
        LOCAL_EMBEDDING_MODEL,
    )


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def list_data_files() -> List[Path]:
    files: List[Path] = []
    for path in DATA_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS or path.suffix.lower() == ".pdf":
            files.append(path)
    return sorted(files)


def read_pdf(path: Path) -> FileContent:
    if PdfReader is None:
        raise RuntimeError("pypdf is required to process PDF files.")
    reader = PdfReader(str(path))
    full_text_parts: List[str] = []
    page_ranges: List[Tuple[int, int, int]] = []
    page_texts: List[str] = []
    cursor = 0
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text and not page_text.endswith("\n"):
            page_text += "\n"
        page_texts.append(page_text)
        start = cursor
        full_text_parts.append(page_text)
        cursor += len(page_text)
        end = cursor
        if end > start:
            page_ranges.append((start, end, i))
    return FileContent(text="".join(full_text_parts), page_ranges=page_ranges, page_texts=page_texts)


def read_file_content(path: Path) -> FileContent:
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return FileContent(text=text, page_ranges=[], page_texts=None)


def build_line_starts(text: str) -> List[int]:
    starts = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            starts.append(idx + 1)
    return starts


def char_to_line(char_pos: int, line_starts: List[int]) -> int:
    if not line_starts:
        return 1
    i = bisect.bisect_right(line_starts, max(char_pos, 0)) - 1
    return max(i + 1, 1)


def char_to_page(char_pos: int, page_ranges: List[Tuple[int, int, int]]) -> Optional[int]:
    if not page_ranges:
        return None
    for start, end, page in page_ranges:
        if start <= char_pos < end:
            return page
    if char_pos >= page_ranges[-1][1]:
        return page_ranges[-1][2]
    return page_ranges[0][2]


def split_with_metadata(
    file_name: str,
    content: FileContent,
    chunk_size: int,
    overlap: int,
    source_path: Optional[str] = None,
) -> List[Document]:
    def split_line_block(lines: List[str], page_num: Optional[int]) -> List[Document]:
        local_docs: List[Document] = []
        seen_ids = set()
        i = 0
        n = len(lines)
        while i < n:
            start_idx = i
            chars = 0
            j = i
            while j < n:
                ln_len = len(lines[j])
                if chars > 0 and chars + ln_len > chunk_size:
                    break
                chars += ln_len
                j += 1
                if chars >= chunk_size:
                    break
            if j <= start_idx:
                j = min(start_idx + 1, n)

            chunk = "".join(lines[start_idx:j])
            if not chunk.strip():
                i = j
                continue

            start_line = start_idx + 1
            end_line = j
            if page_num is None:
                chunk_id = f"{file_name}_{start_line}_{end_line}"
            else:
                # Page-local line numbers for display; page is included in ID to avoid collisions.
                chunk_id = f"{file_name}_p{page_num}_{start_line}_{end_line}"

            if chunk_id in seen_ids:
                i = j
                continue
            seen_ids.add(chunk_id)

            metadata = {
                "file_name": file_name,
                "start_line": start_line,
                "end_line": end_line,
                "chunk_id": chunk_id,
            }
            if source_path:
                metadata["source_path"] = source_path
            if page_num is not None:
                metadata["page"] = page_num
                metadata["page_start"] = page_num
                metadata["page_end"] = page_num

            local_docs.append(Document(page_content=chunk, metadata=metadata))

            if j >= n:
                break

            overlap_chars = 0
            next_i = j
            while next_i > start_idx and overlap_chars < overlap:
                next_i -= 1
                overlap_chars += len(lines[next_i])
            if next_i <= start_idx:
                next_i = min(start_idx + 1, n)
            i = next_i
        return local_docs

    if content.page_texts:
        docs: List[Document] = []
        for page_num, page_text in enumerate(content.page_texts, start=1):
            if not page_text.strip():
                continue
            lines = page_text.splitlines(keepends=True)
            if not lines:
                continue
            docs.extend(split_line_block(lines, page_num))
        return docs

    text = content.text
    if not text.strip():
        return []
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    return split_line_block(lines, page_num=None)


def delete_by_ids(store: Chroma, ids: Iterable[str]) -> None:
    ids = list(ids)
    if ids:
        store.delete(ids=ids)


def process_files(incremental: bool, remove_deleted: bool) -> None:
    ensure_dirs()
    store = get_vector_store()
    manifest = load_manifest()
    manifest_files = manifest.setdefault("files", {})

    current_files = list_data_files()
    current_rel = {str(p.relative_to(DATA_DIR)).replace("\\", "/"): p for p in current_files}
    existing_rel = set(manifest_files.keys())
    removed_rel = existing_rel - set(current_rel.keys())

    total_added = 0
    total_removed = 0
    touched_files = 0

    for rel_name, path in current_rel.items():
        file_hash = sha256_file(path)
        old = manifest_files.get(rel_name)
        changed = (old is None) or (old.get("hash") != file_hash)
        if incremental and not changed:
            continue

        touched_files += 1
        if old and old.get("chunk_ids"):
            delete_by_ids(store, old["chunk_ids"])
            total_removed += len(old["chunk_ids"])

        content = read_file_content(path)
        docs = split_with_metadata(
            file_name=rel_name,
            content=content,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            source_path=str(path.resolve()),
        )
        ids = [d.metadata["chunk_id"] for d in docs]
        if docs:
            store.add_documents(documents=docs, ids=ids)
            total_added += len(docs)

        manifest_files[rel_name] = {
            "hash": file_hash,
            "chunk_ids": ids,
            "updated_at": utc_now_iso(),
        }

    if remove_deleted and removed_rel:
        for rel_name in sorted(removed_rel):
            old = manifest_files.get(rel_name, {})
            old_ids = old.get("chunk_ids", [])
            delete_by_ids(store, old_ids)
            total_removed += len(old_ids)
            manifest_files.pop(rel_name, None)

    save_manifest(manifest)

    mode = "index" if remove_deleted else "add"
    print(f"[{mode}] touched_files={touched_files}, added_chunks={total_added}, removed_chunks={total_removed}")
    print(f"Persisted Chroma DB: {CHROMA_DIR.resolve()}")


def format_sources(docs: List[Document]) -> str:
    lines = []
    seen = set()
    for doc in docs:
        md = doc.metadata
        key = md.get("chunk_id")
        if key in seen:
            continue
        seen.add(key)
        file_name = md.get("file_name", "unknown")
        start_line = md.get("start_line", "?")
        end_line = md.get("end_line", "?")
        page = md.get("page")
        page_end = md.get("page_end")
        lines.append(file_name)
        if page is not None and page_end is not None and page_end != page:
            lines.append(f"ページ: {page}-{page_end}")
        elif page is not None:
            lines.append(f"ページ: {page}")
        lines.append(f"行: {start_line}-{end_line}")
        lines.append("")
    return "\n".join(lines).strip()


def build_context(docs: List[Document], max_chars: int) -> str:
    parts = []
    used = 0
    for d in docs:
        md = d.metadata
        header = (
            f"[file={md.get('file_name')} lines={md.get('start_line')}-{md.get('end_line')}"
            + (f" page={md.get('page')}]\n" if md.get("page") is not None else "]\n")
        )
        body = d.page_content.strip()
        chunk = f"{header}{body}\n\n"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "".join(parts)


def llm_answer(question: str, docs: List[Document]) -> str:
    context = build_context(docs, MAX_CONTEXT_CHARS)
    if not context:
        return "関連するコンテキストが見つかりませんでした。"

    if not USE_OPENAI_CHAT:
        # No chat model configured: return compact extractive answer for stable speed.
        snippets = []
        for d in docs[:3]:
            text = d.page_content.strip().replace("\n", " ")
            snippets.append(text[:220])
        return " / ".join(snippets) if snippets else "関連情報は見つかりましたが要約できませんでした。"

    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, max_tokens=MAX_GENERATION_TOKENS)
    prompt = (
        "あなたは検索結果のみを根拠に回答するアシスタントです。\n"
        "推測は禁止。コンテキストにない内容は『不明』と答える。\n\n"
        f"質問: {question}\n\n"
        f"コンテキスト:\n{context}\n"
    )
    return llm.invoke(prompt).content.strip()


def _tokenize_for_overlap(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+|[ぁ-んァ-ン一-龯ー]+", text.lower())
    expanded: List[str] = []
    for tok in tokens:
        expanded.append(tok)
        # For Japanese tokens, add character bigrams to improve overlap scoring.
        if re.search(r"[ぁ-んァ-ン一-龯ー]", tok) and len(tok) >= 2:
            expanded.extend(tok[i:i + 2] for i in range(len(tok) - 1))
    return expanded


def _lexical_overlap_score(question: str, content: str) -> float:
    q_tokens = set(_tokenize_for_overlap(question))
    if not q_tokens:
        return 0.0
    c_tokens = set(_tokenize_for_overlap(content[:3000]))
    if not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens)
    return overlap / max(len(q_tokens), 1)


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _compact_match_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _longest_common_substring_ratio(query_text: str, content_text: str) -> float:
    if not query_text or not content_text:
        return 0.0
    if query_text in content_text:
        return 1.0
    match = SequenceMatcher(None, query_text, content_text, autojunk=False).find_longest_match(
        0,
        len(query_text),
        0,
        len(content_text),
    )
    return match.size / max(len(query_text), 1)


def search_ranked_matches(
    query_text: str,
    k: int,
    min_score: float = 0.0,
    min_vector_score: float = 0.0,
    min_lexical_score: float = 0.0,
    candidate_multiplier: int = 6,
) -> List[Document]:
    store = get_vector_store()
    fetched = store.get(include=["documents", "metadatas"])
    documents = fetched.get("documents") or []
    metadatas = fetched.get("metadatas") or []
    ids = fetched.get("ids") or []
    if not documents:
        return []

    query_text = query_text.strip()
    if not query_text:
        return []

    query_normalized = _normalize_match_text(query_text)
    query_compact = _compact_match_text(query_text)
    total_docs = len(documents)
    multiplier = max(1, int(candidate_multiplier))
    candidate_k = min(total_docs, max(k, k * multiplier, 50))

    vector_scores: Dict[str, float] = {}
    try:
        pairs = store.similarity_search_with_relevance_scores(query_text, k=candidate_k)
    except Exception:
        pairs = []
    for doc, score in pairs:
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        if not chunk_id:
            continue
        vector_scores[chunk_id] = max(
            vector_scores.get(chunk_id, 0.0),
            max(0.0, min(1.0, float(score))),
        )

    ranked: List[Tuple[float, float, float, float, Document]] = []
    for idx, content in enumerate(documents):
        text = str(content or "")
        if not text.strip():
            continue

        md = dict(metadatas[idx] or {}) if idx < len(metadatas) else {}
        chunk_id = str(md.get("chunk_id") or (ids[idx] if idx < len(ids) else f"chunk_{idx}"))
        md["chunk_id"] = chunk_id

        vector_score = vector_scores.get(chunk_id, 0.0)
        lexical_score = _lexical_overlap_score(query_text, text)
        content_normalized = _normalize_match_text(text)
        content_compact = _compact_match_text(text)
        substring_score = _longest_common_substring_ratio(query_compact, content_compact)

        exact_match = bool(query_normalized) and query_normalized == content_normalized
        phrase_match = bool(query_normalized) and query_normalized in content_normalized
        compact_match = (
            bool(query_compact)
            and len(query_compact) >= 2
            and query_compact in content_compact
        )

        if exact_match:
            match_type = "完全一致"
            final_score = 1.0
        elif phrase_match:
            match_type = "部分一致"
            final_score = min(0.99, 0.95 + 0.03 * substring_score + 0.02 * max(vector_score, lexical_score))
        elif compact_match:
            match_type = "空白差異を無視した一致"
            final_score = min(0.94, 0.90 + 0.05 * substring_score + 0.05 * max(vector_score, lexical_score))
        else:
            match_type = "類似"
            final_score = 0.45 * vector_score + 0.35 * lexical_score + 0.20 * substring_score

        if final_score < min_score:
            continue
        if not (exact_match or phrase_match or compact_match):
            if vector_score < min_vector_score and lexical_score < min_lexical_score:
                continue

        md["match_type"] = match_type
        md["vector_score"] = round(vector_score, 4)
        md["lexical_score"] = round(lexical_score, 4)
        md["substring_score"] = round(substring_score, 4)
        md["relevance_score"] = round(final_score, 4)
        ranked.append(
            (
                final_score,
                substring_score,
                lexical_score,
                vector_score,
                Document(page_content=text, metadata=md),
            )
        )

    ranked.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    return [doc for _, _, _, _, doc in ranked[:k]]


def search_relevant_docs(
    question: str,
    k: int,
    relevance_threshold: Optional[float] = None,
    min_vector_score: Optional[float] = None,
    min_lexical_score: Optional[float] = None,
    candidate_multiplier: Optional[int] = None,
) -> List[Document]:
    store = get_vector_store()
    threshold = RELEVANCE_SCORE_THRESHOLD if relevance_threshold is None else relevance_threshold
    min_vec = MIN_VECTOR_SCORE if min_vector_score is None else min_vector_score
    min_lex = MIN_LEXICAL_SCORE if min_lexical_score is None else min_lexical_score
    multiplier = CANDIDATE_MULTIPLIER if candidate_multiplier is None else max(1, int(candidate_multiplier))
    candidate_k = max(k, k * multiplier)
    pairs = store.similarity_search_with_relevance_scores(question, k=candidate_k)
    if not pairs:
        return []

    rescored: List[Tuple[float, float, float, Document]] = []
    for doc, score in pairs:
        vec_score = max(0.0, min(1.0, float(score)))
        lex_score = _lexical_overlap_score(question, doc.page_content)
        final = 0.7 * vec_score + 0.3 * lex_score
        new_md = dict(doc.metadata)
        new_md["vector_score"] = round(vec_score, 4)
        new_md["lexical_score"] = round(lex_score, 4)
        new_md["relevance_score"] = round(final, 4)
        rescored.append((final, vec_score, lex_score, Document(page_content=doc.page_content, metadata=new_md)))

    rescored.sort(key=lambda x: x[0], reverse=True)
    filtered = [
        d
        for final, vec, lex, d in rescored
        if final >= threshold and vec >= min_vec and lex >= min_lex
    ]
    return filtered[:k]


def query(question: str, k: int) -> None:
    ensure_dirs()
    docs = search_relevant_docs(question, k=k)
    if not docs:
        print("【回答】")
        print("該当情報が見つかりませんでした。")
        print("\n【出典】")
        print("(なし)")
        return

    answer = llm_answer(question, docs)
    sources = format_sources(docs)
    print("【回答】")
    print(answer)
    print("\n【出典】")
    print(sources)


def delete_files(file_names: List[str]) -> None:
    ensure_dirs()
    store = get_vector_store()
    manifest = load_manifest()
    manifest_files = manifest.setdefault("files", {})

    removed = 0
    for name in file_names:
        normalized = name.replace("\\", "/")
        old = manifest_files.get(normalized)
        if not old:
            print(f"[delete] skip (not found in manifest): {normalized}")
            continue
        ids = old.get("chunk_ids", [])
        delete_by_ids(store, ids)
        removed += len(ids)
        manifest_files.pop(normalized, None)
        print(f"[delete] removed file index: {normalized} ({len(ids)} chunks)")

    save_manifest(manifest)
    print(f"[delete] total_removed_chunks={removed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local RAG system with persistent Chroma and incremental indexing."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="Full sync index: add/update files and remove deleted files.")
    sub.add_parser("add", help="Incremental add/update only (do not remove missing files).")

    query_parser = sub.add_parser("query", help="Query the RAG index.")
    query_parser.add_argument("question", type=str, help='Question text, e.g. "仕様は何ですか？"')
    query_parser.add_argument("--k", type=int, default=TOP_K, help=f"Top-k retrieval (default: {TOP_K})")

    delete_parser = sub.add_parser("delete", help="Delete indexed file(s) from vector store.")
    delete_parser.add_argument("files", nargs="+", help="Relative paths under data/, e.g. manual_A.txt")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        process_files(incremental=True, remove_deleted=True)
    elif args.command == "add":
        process_files(incremental=True, remove_deleted=False)
    elif args.command == "query":
        query(question=args.question, k=args.k)
    elif args.command == "delete":
        delete_files(args.files)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
