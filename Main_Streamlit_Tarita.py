import hashlib
import html
import io
import json
import os
import base64
import csv
import gc
import re
import zipfile
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import streamlit as st

import app

PATH_SETTINGS_FILE = Path(__file__).with_name("path_settings.json")
LANGUAGE_OPTIONS = [("日本語", "ja"), ("English", "en")]
LANGUAGE_LABEL_TO_CODE = {label: code for label, code in LANGUAGE_OPTIONS}
LANGUAGE_CODE_TO_LABEL = {code: label for label, code in LANGUAGE_OPTIONS}
ENGLISH_DATA_DIRNAME = "en"
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "manual_rag")


def _get_upload_cache_dir() -> Path:
    return app.DATA_DIR / "streamlit_uploads"


def _build_language_runtime(
    data_dir: str,
    chroma_dir: str,
    language_code: str,
) -> Tuple[Path, Path, Path, str, set]:
    base_collection_name = getattr(
        app,
        "BASE_COLLECTION_NAME",
        getattr(app, "COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
    )
    data_root = Path(data_dir).expanduser().resolve()
    chroma_root = Path(chroma_dir).expanduser().resolve()
    if language_code == "en":
        data_path = data_root / ENGLISH_DATA_DIRNAME
        manifest_path = chroma_root / "index_manifest_en.json"
        collection_name = f"{base_collection_name}_en"
        ignored_dirs = set()
    else:
        # Keep existing Japanese data/index visible by default.
        data_path = data_root
        manifest_path = chroma_root / "index_manifest.json"
        collection_name = base_collection_name
        ignored_dirs = {ENGLISH_DATA_DIRNAME}
    return data_path, chroma_root, manifest_path, collection_name, ignored_dirs


def _load_path_settings() -> Tuple[str, str]:
    default_data = str(app.DATA_DIR.resolve())
    default_chroma = str(app.CHROMA_DIR.resolve())
    if not PATH_SETTINGS_FILE.exists():
        return default_data, default_chroma
    try:
        data = json.loads(PATH_SETTINGS_FILE.read_text(encoding="utf-8"))
        data_dir = str(Path(data.get("data_dir", default_data)).expanduser())
        chroma_dir = str(Path(data.get("chroma_dir", default_chroma)).expanduser())
        return data_dir, chroma_dir
    except Exception:
        return default_data, default_chroma


def _save_path_settings(data_dir: str, chroma_dir: str) -> None:
    payload = {"data_dir": data_dir, "chroma_dir": chroma_dir}
    PATH_SETTINGS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pick_folder_dialog(initial_dir: str) -> Optional[str]:
    if os.name != "nt":
        return None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        initial = initial_dir if Path(initial_dir).exists() else str(Path.home())
        selected = filedialog.askdirectory(initialdir=initial, title="フォルダを選択")
        root.destroy()
        if not selected:
            return None
        return str(Path(selected).resolve())
    except Exception:
        return None


def _apply_runtime_paths(data_dir: str, chroma_dir: str, language_code: str) -> Tuple[Path, Path]:
    data_path, chroma_path, manifest_path, collection_name, ignored_dirs = _build_language_runtime(
        data_dir,
        chroma_dir,
        language_code,
    )
    data_path.mkdir(parents=True, exist_ok=True)
    chroma_path.mkdir(parents=True, exist_ok=True)
    app.DATA_DIR = data_path
    app.CHROMA_DIR = chroma_path
    app.MANIFEST_PATH = manifest_path
    app.COLLECTION_NAME = collection_name
    app.ACTIVE_LANGUAGE = language_code
    app.IGNORED_TOP_LEVEL_DIRS = ignored_dirs
    return data_path, chroma_path


def _render_match_excerpt(text: str) -> None:
    excerpt = text.strip() or "(テキストなし)"
    if getattr(app, "ACTIVE_LANGUAGE", "") != "en":
        st.code(excerpt, language="text")
        return
    escaped = html.escape(excerpt)
    st.markdown(
        f"""
        <div style="
            font-family: 'Times New Roman', Times, serif;
            white-space: pre-wrap;
            line-height: 1.65;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.5rem;
            padding: 0.9rem 1rem;
            background: rgba(250, 250, 250, 0.9);
        ">{escaped}</div>
        """,
        unsafe_allow_html=True,
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_key(name: str) -> str:
    return name.replace("\\", "/")


def _is_excluded_from_folder_index(root: Path, target: Path) -> bool:
    ignored_dirs = getattr(app, "IGNORED_TOP_LEVEL_DIRS", set())
    if not ignored_dirs:
        return False
    try:
        rel = target.relative_to(root)
    except ValueError:
        return False
    return bool(rel.parts) and rel.parts[0] in ignored_dirs


def _uploaded_size(uploaded_file) -> int:
    size = getattr(uploaded_file, "size", None)
    if size is not None:
        return int(size)
    try:
        return len(uploaded_file.getbuffer())
    except Exception:
        return len(uploaded_file.getvalue() or b"")


def _parse_pdf_password_candidates(raw_text: str) -> List[str]:
    candidates: List[str] = []
    for line in str(raw_text or "").splitlines():
        password = line.strip()
        if password and password not in candidates:
            candidates.append(password)
    return candidates


def _verify_chroma_chunk_ids(store, chunk_ids: List[str]) -> None:
    if not chunk_ids:
        return
    try:
        fetched = store.get(ids=chunk_ids, include=["metadatas"])
    except Exception as e:
        raise RuntimeError(f"Chroma保存確認に失敗しました: {e}") from e
    existing_ids = {str(chunk_id) for chunk_id in (fetched.get("ids") or [])}
    if len(existing_ids) < len(chunk_ids):
        missing = len(chunk_ids) - len(existing_ids)
        raise RuntimeError(f"Chroma保存確認に失敗しました。{missing} 件のチャンクが見つかりません。")


# ── ローカルバックアップ / サーバー上データ削除 ─────────────────────────────


def _zip_directory(root_dir: Path) -> bytes:
    root_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(root_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(root_dir))
    return buf.getvalue()


def _safe_extract_zip(zip_file: zipfile.ZipFile, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    for member in zip_file.infolist():
        member_name = member.filename
        if not member_name:
            continue
        target_path = (dest_dir / member_name).resolve()
        if target_path != dest_root and dest_root not in target_path.parents:
            raise ValueError("zip内に不正なパスが含まれています")
    zip_file.extractall(dest_dir)


def _remove_tree_contents(root_dir: Path) -> int:
    if not root_dir.exists():
        return 0
    removed = 0
    for path in sorted(root_dir.iterdir(), key=lambda p: (p.is_file(), len(p.parts)), reverse=True):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        removed += 1
    return removed


def _backup_file_name() -> str:
    return f"manualsearch_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"


def has_local_backup_source() -> bool:
    if not app.CHROMA_DIR.exists():
        return False
    return any(path.is_file() for path in app.CHROMA_DIR.rglob("*"))


def build_local_backup_zip() -> Tuple[Optional[bytes], str]:
    app.ensure_dirs()
    if not has_local_backup_source():
        return None, "保存対象のインデックスがありません"
    zip_data = _zip_directory(app.CHROMA_DIR)
    return zip_data, f"バックアップ準備完了 ({len(zip_data) // 1024} KB)"


def restore_index_from_local_backup(uploaded_file) -> Tuple[bool, str]:
    if uploaded_file is None:
        return False, "バックアップzipが選択されていません"
    try:
        zip_bytes = uploaded_file.getvalue()
    except Exception:
        zip_bytes = b""
    if not zip_bytes:
        return False, "バックアップzipが空です"

    try:
        app.reset_chroma_system_cache()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            members = [name for name in zf.namelist() if name and not name.endswith("/")]
            if not members:
                return False, "zip内に復元対象ファイルがありません"
            _remove_tree_contents(app.CHROMA_DIR)
            app.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            _safe_extract_zip(zf, app.CHROMA_DIR)
        app.reset_chroma_system_cache()
    except zipfile.BadZipFile:
        return False, "zipファイルとして読み込めませんでした"
    except Exception as e:
        return False, f"復元に失敗しました: {e}"

    restored_manifest = app.load_manifest()
    restored_files = len(restored_manifest.get("files", {}))
    return True, f"ローカルバックアップを復元しました ({len(zip_bytes) // 1024} KB / 登録ファイル数={restored_files})"


def clear_server_runtime_data(data_root_dir: str, chroma_root_dir: str) -> Tuple[bool, str]:
    data_root = Path(data_root_dir).expanduser().resolve()
    chroma_root = Path(chroma_root_dir).expanduser().resolve()
    removed_items = 0

    app.reset_chroma_system_cache()

    upload_dirs = [
        data_root / "streamlit_uploads",
        data_root / ENGLISH_DATA_DIRNAME / "streamlit_uploads",
    ]
    for upload_dir in upload_dirs:
        removed_items += _remove_tree_contents(upload_dir)
        try:
            if upload_dir.exists():
                upload_dir.rmdir()
        except OSError:
            pass

    removed_items += _remove_tree_contents(chroma_root)
    chroma_root.mkdir(parents=True, exist_ok=True)
    app.ensure_dirs()
    app.save_manifest({"files": {}})
    app.reset_chroma_system_cache()
    return True, f"サーバー上データを削除しました (削除対象={removed_items}件)"


# ─────────────────────────────────────────────────────────────────────────────


def _upsert_pdf(
    logical_file_name: str,
    file_path: Path,
    file_hash: str,
    store,
    manifest_files: Dict,
    stage_cb: Optional[Callable[[str], None]] = None,
    pdf_password_candidates: Optional[List[str]] = None,
) -> Tuple[str, int, int]:
    logical_file_name = _normalize_key(logical_file_name)
    old = manifest_files.get(logical_file_name)
    old_chunk_ids = list(old.get("chunk_ids") or []) if old else []
    same_hash = old is not None and old.get("hash") == file_hash
    missing_old_chunks = False
    if old is not None:
        if not old_chunk_ids:
            missing_old_chunks = True
        else:
            try:
                fetched = store.get(ids=old_chunk_ids, include=["metadatas"])
                existing_ids = {str(chunk_id) for chunk_id in (fetched.get("ids") or [])}
                missing_old_chunks = len(existing_ids) < len(old_chunk_ids)
            except Exception:
                missing_old_chunks = True

    changed = (old is None) or (not same_hash) or missing_old_chunks
    if not changed:
        return "skipped", 0, 0

    if stage_cb:
        stage_cb("PDFテキスト抽出中...")
    content = app.read_pdf(file_path, passwords=pdf_password_candidates)
    if stage_cb:
        stage_cb("チャンク分割中...")
    docs = app.split_with_metadata(
        file_name=logical_file_name,
        content=content,
        chunk_size=app.CHUNK_SIZE,
        overlap=app.CHUNK_OVERLAP,
        source_path=str(file_path.resolve()),
    )
    if not docs:
        raise RuntimeError("PDFからテキストを抽出できませんでした。画像PDFまたは抽出不能PDFの可能性があります。")

    ids = [d.metadata["chunk_id"] for d in docs]

    removed = 0
    if old_chunk_ids:
        if stage_cb:
            stage_cb("既存チャンクを削除中...")
        app.delete_by_ids(store, old_chunk_ids)
        removed = len(old_chunk_ids)

    if docs:
        if stage_cb:
            stage_cb("ベクトル化してChromaへ保存中...")
        store.add_documents(docs, ids=ids)
        if stage_cb:
            stage_cb("Chroma保存確認中...")
        _verify_chroma_chunk_ids(store, ids)

    manifest_files[logical_file_name] = {
        "hash": file_hash,
        "chunk_ids": ids,
        "updated_at": _utc_now_iso(),
    }
    action = "added" if old is None else "updated"
    return action, len(ids), removed


def index_uploaded_pdfs(
    uploaded_files,
    progress_cb: Optional[Callable[[float, str, int, int, int], None]] = None,
    pdf_password_candidates: Optional[List[str]] = None,
) -> Tuple[int, int, int, List[str], Dict[str, object]]:
    total = max(len(uploaded_files), 1)

    def emit(progress: float, text: str, t: int, a: int, r: int) -> None:
        if progress_cb:
            progress_cb(max(0.0, min(1.0, progress)), text, t, a, r)

    emit(0.01, "初期化中...", 0, 0, 0)
    app.ensure_dirs()
    upload_cache_dir = _get_upload_cache_dir()
    upload_cache_dir.mkdir(parents=True, exist_ok=True)
    emit(0.08, "埋め込みモデル / ベクトルストア準備中...", 0, 0, 0)
    store = app.get_vector_store()
    emit(0.15, "インデックス状態の読み込み中...", 0, 0, 0)
    manifest = app.load_manifest()
    manifest_files = manifest.setdefault("files", {})
    before_file_count = len(manifest_files)

    touched = 0
    added = 0
    removed = 0
    errors: List[str] = []
    added_files: List[str] = []
    updated_files: List[str] = []
    skipped_files: List[str] = []

    for idx, uploaded in enumerate(uploaded_files, start=1):
        file_start = 0.15 + 0.8 * ((idx - 1) / total)
        file_span = 0.8 / total
        emit(file_start, f"{uploaded.name}: 取り込み準備中...", touched, added, removed)
        try:
            save_path = upload_cache_dir / uploaded.name
            data_view = None
            try:
                data_view = uploaded.getbuffer()
                if len(data_view) == 0:
                    emit(file_start + file_span, f"{uploaded.name}: 空ファイルのためスキップ", touched, added, removed)
                    continue
                save_path.write_bytes(data_view)
                file_hash = hashlib.sha256(data_view).hexdigest()
            finally:
                if data_view is not None:
                    try:
                        data_view.release()
                    except Exception:
                        pass

            logical_name = f"streamlit_uploads/{uploaded.name}"
            stage_ratio = {
                "既存チャンクを削除中...": 0.20,
                "PDFテキスト抽出中...": 0.45,
                "チャンク分割中...": 0.70,
                "ベクトル化してChromaへ保存中...": 0.90,
                "Chroma保存確認中...": 0.96,
            }

            def on_stage(msg: str, name: str = uploaded.name) -> None:
                ratio = stage_ratio.get(msg, 0.5)
                emit(file_start + file_span * ratio, f"{name}: {msg}", touched, added, removed)

            action, a, r = _upsert_pdf(
                logical_file_name=logical_name,
                file_path=save_path,
                file_hash=file_hash,
                store=store,
                manifest_files=manifest_files,
                stage_cb=on_stage if progress_cb else None,
                pdf_password_candidates=pdf_password_candidates,
            )
            if action != "skipped":
                touched += 1
                added += a
                removed += r
            if action == "added":
                added_files.append(logical_name)
            elif action == "updated":
                updated_files.append(logical_name)
            else:
                skipped_files.append(logical_name)
            app.save_manifest(manifest)
            gc.collect()
            emit(file_start + file_span, f"{uploaded.name}: 完了", touched, added, removed)
        except Exception as e:
            errors.append(f"{uploaded.name}: {e}")
            app.save_manifest(manifest)
            gc.collect()
            emit(file_start + file_span, f"{uploaded.name}: エラーのためスキップ", touched, added, removed)

    emit(0.97, "結果保存中...", touched, added, removed)
    app.save_manifest(manifest)
    final_text = "取り込み完了" if not errors else f"取り込み完了（一部エラー {len(errors)} 件）"
    emit(1.0, final_text, touched, added, removed)
    summary = {
        "before_file_count": before_file_count,
        "after_file_count": len(manifest_files),
        "added_files": added_files,
        "updated_files": updated_files,
        "skipped_files": skipped_files,
    }
    return touched, added, removed, errors, summary


def _uploaded_signature(uploaded_files) -> str:
    if not uploaded_files:
        return ""
    parts = []
    for f in uploaded_files:
        parts.append(f"{f.name}:{_uploaded_size(f)}")
    return "|".join(sorted(parts))


def index_pdf_folder(
    folder_path: str,
    progress_cb: Optional[Callable[[float, str, int, int, int], None]] = None,
    pdf_password_candidates: Optional[List[str]] = None,
) -> Tuple[int, int, int, int]:
    def emit(progress: float, text: str, t: int, a: int, r: int) -> None:
        if progress_cb:
            progress_cb(max(0.0, min(1.0, progress)), text, t, a, r)

    emit(0.01, "初期化中...", 0, 0, 0)
    app.ensure_dirs()
    emit(0.08, "埋め込みモデル / ベクトルストア準備中...", 0, 0, 0)
    store = app.get_vector_store()
    emit(0.15, "インデックス状態の読み込み中...", 0, 0, 0)
    manifest = app.load_manifest()
    manifest_files = manifest.setdefault("files", {})

    root = Path(folder_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")

    pdf_files = sorted(
        p
        for p in root.rglob("*.pdf")
        if p.is_file() and not _is_excluded_from_folder_index(root, p)
    )
    touched = 0
    added = 0
    removed = 0

    root_label = root.name
    total = len(pdf_files)
    if total == 0:
        emit(1.0, "PDFが見つかりませんでした", touched, added, removed)
        app.save_manifest(manifest)
        return 0, touched, added, removed
    for idx, pdf in enumerate(pdf_files, start=1):
        file_start = 0.15 + 0.8 * ((idx - 1) / total)
        file_span = 0.8 / total
        rel = str(pdf.relative_to(root)).replace("\\", "/")
        logical_name = f"folder/{root_label}/{rel}"
        emit(file_start, f"{rel}: 取り込み準備中...", touched, added, removed)

        stage_ratio = {
            "既存チャンクを削除中...": 0.20,
            "PDFテキスト抽出中...": 0.45,
            "チャンク分割中...": 0.70,
            "ベクトル化してChromaへ保存中...": 0.90,
            "Chroma保存確認中...": 0.96,
        }

        def on_stage(msg: str, rel_name: str = rel) -> None:
            ratio = stage_ratio.get(msg, 0.5)
            emit(file_start + file_span * ratio, f"{rel_name}: {msg}", touched, added, removed)

        changed, a, r = _upsert_pdf(
            logical_file_name=logical_name,
            file_path=pdf,
            file_hash=app.sha256_file(pdf),
            store=store,
            manifest_files=manifest_files,
            stage_cb=on_stage if progress_cb else None,
            pdf_password_candidates=pdf_password_candidates,
        )
        if changed:
            touched += 1
            added += a
            removed += r
        app.save_manifest(manifest)
        gc.collect()
        emit(file_start + file_span, f"{rel}: 完了", touched, added, removed)

    emit(0.97, "結果保存中...", touched, added, removed)
    app.save_manifest(manifest)
    emit(1.0, "取り込み完了", touched, added, removed)
    return len(pdf_files), touched, added, removed


def run_query(
    search_text: str,
    k: int,
    min_score: float,
    min_vector_score: float,
    min_lexical_score: float,
    candidate_multiplier: int,
) -> List:
    return app.search_ranked_matches(
        search_text,
        k=k,
        min_score=min_score,
        min_vector_score=min_vector_score,
        min_lexical_score=min_lexical_score,
        candidate_multiplier=candidate_multiplier,
    )


def _resolve_source_path(md: Dict) -> Optional[Path]:
    source_path = md.get("source_path")
    if source_path:
        p = Path(source_path)
        if p.exists():
            return p

    file_name = md.get("file_name")
    if not file_name:
        return None

    # Fallback: rebuild from managed relative key in manifest metadata.
    rel = file_name.replace("\\", "/")
    if rel.startswith("streamlit_uploads/"):
        p = app.DATA_DIR / rel
        if p.exists():
            return p.resolve()

    if rel.startswith("folder/"):
        # If this came from folder ingest, source_path should exist.
        # Keep best-effort fallback for users who manually moved files.
        candidate = Path(rel.split("/", 2)[-1])
        if candidate.exists():
            return candidate.resolve()

    p = app.DATA_DIR / rel
    if p.exists():
        return p.resolve()
    return None


def _render_pdf_inline(path_obj: Path, page: Optional[int] = None) -> None:
    data = path_obj.read_bytes()
    if len(data) > 25 * 1024 * 1024:
        st.warning("PDFが大きいため、アプリ内表示が重い可能性があります。OSで開くを推奨します。")
    b64 = base64.b64encode(data).decode("utf-8")
    page_no = page if page is not None else 1
    src = f"data:application/pdf;base64,{b64}#page={page_no}"
    iframe = (
        f'<iframe src="{src}" width="100%" height="700" '
        f'style="border:1px solid #ddd; border-radius:8px;"></iframe>'
    )
    st.components.v1.html(iframe, height=720, scrolling=True)


def render_sources(docs: List) -> None:
    if not docs:
        st.info("一致または近い箇所が見つかりませんでした。")
        return
    for idx, d in enumerate(docs):
        md = d.metadata
        file_name = md.get("file_name", "unknown")
        start_line = md.get("start_line", "?")
        end_line = md.get("end_line", "?")
        page = md.get("page_start", md.get("page"))
        page_end = md.get("page_end", page)
        source_path = md.get("source_path")
        match_type = md.get("match_type", "類似")
        score = md.get("relevance_score")
        if page is not None and page_end is not None and page_end != page:
            page_text = f"{page}-{page_end}"
        elif page is not None:
            page_text = str(page)
        else:
            page_text = "(不明)"

        if score is None:
            expander_title = f"{idx + 1}. {match_type} / {file_name} / ページ:{page_text} / 行:{start_line}-{end_line}"
        else:
            expander_title = (
                f"{idx + 1}. {match_type} / {file_name} / ページ:{page_text} / "
                f"行:{start_line}-{end_line} / score:{score}"
            )
        with st.expander(expander_title, expanded=(idx == 0)):
            st.markdown(f"- 順位: `{idx + 1}`")
            st.markdown(f"- 一致種別: `{match_type}`")
            st.markdown(f"- ファイル: `{file_name}`")
            st.markdown(f"- ページ: `{page_text}`")
            st.markdown(f"- 行(ページ内): `{start_line}-{end_line}`")
            if score is not None:
                st.markdown(f"- 最終スコア: `{score}`")
            vector_score = md.get("vector_score")
            if vector_score is not None:
                st.markdown(f"- ベクトル類似度: `{vector_score}`")
            lexical_score = md.get("lexical_score")
            if lexical_score is not None:
                st.markdown(f"- 語彙一致率: `{lexical_score}`")
            substring_score = md.get("substring_score")
            if substring_score is not None:
                st.markdown(f"- 連続一致率: `{substring_score}`")

            path_obj = _resolve_source_path(md)
            if path_obj:
                if path_obj.exists():
                    st.code(str(path_obj), language="text")
                else:
                    st.markdown("- ファイルパス: (存在しません)")
            elif source_path:
                st.markdown("- ファイルパス: (存在しません)")
                st.code(str(source_path), language="text")
            else:
                st.markdown("- ファイルパス: (未記録。再インデックスで改善します)")

            st.markdown("**該当箇所**")
            _render_match_excerpt(d.page_content)


def delete_indexed_file(file_name: str) -> int:
    manifest = app.load_manifest()
    manifest_files = manifest.setdefault("files", {})
    old = manifest_files.get(file_name)
    if not old:
        return 0
    ids = old.get("chunk_ids", [])
    store = app.get_vector_store()
    app.delete_by_ids(store, ids)
    manifest_files.pop(file_name, None)
    app.save_manifest(manifest)
    return len(ids)


def get_indexed_file_names() -> List[str]:
    manifest = app.load_manifest()
    return sorted(manifest.get("files", {}).keys())


def _normalize_debug_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _build_debug_hit(md: Dict, text: str, position: int) -> Dict[str, object]:
    file_name = str(md.get("file_name", "unknown"))
    page = md.get("page_start", md.get("page"))
    start_line = md.get("start_line", "?")
    end_line = md.get("end_line", "?")
    snippet_start = max(0, position - 80)
    snippet_end = min(len(text), position + 180)
    snippet = text[snippet_start:snippet_end].strip()
    return {
        "file_name": file_name,
        "page": page,
        "start_line": start_line,
        "end_line": end_line,
        "snippet": snippet,
        "chunk_id": str(md.get("chunk_id", "")),
    }


def debug_search_stored_chunks(search_text: str, max_hits: int = 20) -> Dict[str, object]:
    query_text = search_text.strip()
    if not query_text:
        return {
            "total_chunks": 0,
            "raw_hits": [],
            "normalized_hits": [],
        }

    store = app.get_vector_store()
    fetched = store.get(include=["documents", "metadatas"])
    documents = fetched.get("documents") or []
    metadatas = fetched.get("metadatas") or []
    raw_hits: List[Dict[str, object]] = []
    normalized_hits: List[Dict[str, object]] = []
    raw_hit_ids = set()
    normalized_query = _normalize_debug_text(query_text)

    for idx, content in enumerate(documents):
        text = str(content or "")
        if not text:
            continue
        md = dict(metadatas[idx] or {}) if idx < len(metadatas) else {}
        chunk_id = str(md.get("chunk_id") or f"chunk_{idx}")
        md["chunk_id"] = chunk_id

        raw_pos = text.find(query_text)
        if raw_pos >= 0 and len(raw_hits) < max_hits:
            raw_hits.append(_build_debug_hit(md, text, raw_pos))
            raw_hit_ids.add(chunk_id)

        normalized_text = _normalize_debug_text(text)
        normalized_pos = normalized_text.find(normalized_query)
        if normalized_pos >= 0 and chunk_id not in raw_hit_ids and len(normalized_hits) < max_hits:
            normalized_hits.append(_build_debug_hit(md, text, 0))

    return {
        "total_chunks": len(documents),
        "raw_hits": raw_hits,
        "normalized_hits": normalized_hits,
    }


def get_indexed_file_chunk_stats() -> Dict[str, object]:
    manifest = app.load_manifest()
    manifest_files = manifest.get("files", {})
    store = app.get_vector_store()
    fetched = store.get(include=["metadatas"])
    metadatas = fetched.get("metadatas") or []

    chroma_counts: Dict[str, int] = {}
    for md in metadatas:
        file_name = str((md or {}).get("file_name", "")).strip()
        if not file_name:
            continue
        chroma_counts[file_name] = chroma_counts.get(file_name, 0) + 1

    all_names = sorted(set(manifest_files.keys()) | set(chroma_counts.keys()))
    rows: List[Dict[str, object]] = []
    for file_name in all_names:
        manifest_chunk_ids = list((manifest_files.get(file_name) or {}).get("chunk_ids") or [])
        manifest_chunks = len(manifest_chunk_ids)
        chroma_chunks = int(chroma_counts.get(file_name, 0))
        if manifest_chunks == chroma_chunks and chroma_chunks > 0:
            status = "OK"
        elif manifest_chunks == 0 and chroma_chunks == 0:
            status = "未登録"
        elif chroma_chunks == 0:
            status = "Chroma未保存"
        elif manifest_chunks == 0:
            status = "Manifest未登録"
        else:
            status = "不一致"
        rows.append(
            {
                "status": status,
                "file_name": file_name,
                "manifest_chunks": manifest_chunks,
                "chroma_chunks": chroma_chunks,
            }
        )

    return {
        "rows": rows,
        "manifest_file_count": len(manifest_files),
        "chroma_file_count": len(chroma_counts),
        "chroma_chunk_total": sum(chroma_counts.values()),
        "ok_files": sum(1 for row in rows if row["status"] == "OK"),
        "problem_files": sum(1 for row in rows if row["status"] != "OK"),
    }


def _parse_optional_int(value: object) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _normalize_eval_file_name(name: object) -> str:
    return _normalize_key(str(name or "").strip()).lower()


def _file_name_matches(actual_name: object, expected_name: object) -> bool:
    actual = _normalize_eval_file_name(actual_name)
    expected = _normalize_eval_file_name(expected_name)
    if not actual or not expected:
        return False
    if actual == expected:
        return True
    return Path(actual).name == Path(expected).name


def _doc_matches_expected(doc, expected_file: str, expected_page: Optional[int], expected_line: Optional[int]) -> bool:
    md = doc.metadata or {}
    if not _file_name_matches(md.get("file_name", ""), expected_file):
        return False

    if expected_page is not None:
        page_start = _parse_optional_int(md.get("page_start", md.get("page")))
        page_end = _parse_optional_int(md.get("page_end", md.get("page")))
        if page_start is None:
            return False
        if page_end is None:
            page_end = page_start
        if not (page_start <= expected_page <= page_end):
            return False

    if expected_line is not None:
        line_start = _parse_optional_int(md.get("start_line"))
        line_end = _parse_optional_int(md.get("end_line", md.get("start_line")))
        if line_start is None:
            return False
        if line_end is None:
            line_end = line_start
        if not (line_start <= expected_line <= line_end):
            return False

    return True


def _decode_csv_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, "UTF-8/CP932 として読み込めませんでした")


def _pick_csv_field(field_map: Dict[str, str], *candidates: str) -> Optional[str]:
    for candidate in candidates:
        if candidate in field_map:
            return field_map[candidate]
    return None


def _parse_evaluation_cases(uploaded_file) -> Tuple[List[Dict[str, object]], List[str]]:
    try:
        csv_bytes = uploaded_file.getvalue()
    except Exception:
        csv_bytes = b""
    if not csv_bytes:
        raise ValueError("評価CSVが空です。")

    decoded = _decode_csv_bytes(csv_bytes)
    sample = decoded[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","
    reader = csv.DictReader(io.StringIO(decoded), delimiter=delimiter, skipinitialspace=True)
    if not reader.fieldnames:
        raise ValueError("CSVのヘッダー行が見つかりません。")

    field_map = {str(name or "").strip().lower(): str(name) for name in reader.fieldnames if str(name or "").strip()}
    query_key = _pick_csv_field(field_map, "query", "search_text", "search", "text")
    expected_file_key = _pick_csv_field(field_map, "expected_file", "file_name", "expected_filename", "filename")
    expected_page_key = _pick_csv_field(field_map, "expected_page", "page", "page_no")
    expected_line_key = _pick_csv_field(field_map, "expected_line", "line", "start_line")
    notes_key = _pick_csv_field(field_map, "notes", "note", "memo", "comment")

    if not query_key or not expected_file_key:
        raise ValueError("CSVには少なくとも 'query' 列と 'expected_file' 列が必要です。")

    cases: List[Dict[str, object]] = []
    parse_errors: List[str] = []
    for row_index, row in enumerate(reader, start=2):
        values = [str(v or "").strip() for v in row.values()]
        if not any(values):
            continue

        query_text = str(row.get(query_key) or "").strip()
        expected_file = str(row.get(expected_file_key) or "").strip()
        expected_page_raw = str(row.get(expected_page_key) or "").strip() if expected_page_key else ""
        expected_line_raw = str(row.get(expected_line_key) or "").strip() if expected_line_key else ""
        notes = str(row.get(notes_key) or "").strip() if notes_key else ""

        if not query_text:
            parse_errors.append(f"{row_index}行目: query が空です。")
            continue
        if not expected_file:
            parse_errors.append(f"{row_index}行目: expected_file が空です。")
            continue

        expected_page = _parse_optional_int(expected_page_raw)
        expected_line = _parse_optional_int(expected_line_raw)
        if expected_page_raw and expected_page is None:
            parse_errors.append(f"{row_index}行目: expected_page が数値ではありません。")
            continue
        if expected_line_raw and expected_line is None:
            parse_errors.append(f"{row_index}行目: expected_line が数値ではありません。")
            continue

        cases.append(
            {
                "row_no": row_index,
                "query": query_text,
                "expected_file": expected_file,
                "expected_page": expected_page,
                "expected_line": expected_line,
                "notes": notes,
            }
        )
    return cases, parse_errors


def _doc_result_summary(doc) -> Dict[str, object]:
    md = doc.metadata or {}
    detected_text = re.sub(r"\s+", " ", str(getattr(doc, "page_content", "") or "")).strip()
    return {
        "file": str(md.get("file_name", "")),
        "page": _parse_optional_int(md.get("page_start", md.get("page"))),
        "line": _parse_optional_int(md.get("start_line")),
        "score": md.get("relevance_score", ""),
        "match_type": str(md.get("match_type", "")),
        "text": detected_text,
    }


def _build_evaluation_csv(rows: List[Dict[str, object]]) -> bytes:
    if not rows:
        return b""
    header = [
        "query",
        "expected_file",
        "expected_page",
        "expected_line",
        "notes",
        "row_no",
        "status",
        "first_match_rank",
        "top1_hit",
        "top3_hit",
        "detected_file",
        "detected_page",
        "detected_line",
        "detected_score",
        "detected_type",
        "detected_text",
        "top1_file",
        "top1_page",
        "top1_line",
        "top1_score",
        "top1_type",
        "top1_text",
        "error",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in header})
    return buf.getvalue().encode("utf-8-sig")


def _evaluation_file_name(language_code: str) -> str:
    suffix = "en" if language_code == "en" else "ja"
    return f"manualsearch_evaluation_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def run_evaluation_cases(
    uploaded_file,
    top_k: int,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, object]:
    cases, parse_errors = _parse_evaluation_cases(uploaded_file)
    rows: List[Dict[str, object]] = []
    total = len(cases)
    top1_hits = 0
    top3_hits = 0
    matched_total = 0
    execution_errors = 0
    top3_window = min(3, max(1, int(top_k)))

    for idx, case in enumerate(cases, start=1):
        query_text = str(case["query"])
        if progress_cb:
            progress_cb(idx, total, query_text)

        try:
            results = app.search_ranked_matches(
                query_text,
                k=int(top_k),
                min_score=0.0,
                min_vector_score=0.0,
                min_lexical_score=0.0,
                candidate_multiplier=app.CANDIDATE_MULTIPLIER,
            )
        except Exception as e:
            execution_errors += 1
            rows.append(
                {
                    "query": query_text,
                    "expected_file": case["expected_file"],
                    "expected_page": case["expected_page"] or "",
                    "expected_line": case["expected_line"] or "",
                    "notes": case["notes"],
                    "row_no": case["row_no"],
                    "status": "error",
                    "first_match_rank": "",
                    "top1_hit": "",
                    "top3_hit": "",
                    "detected_file": "",
                    "detected_page": "",
                    "detected_line": "",
                    "detected_score": "",
                    "detected_type": "",
                    "detected_text": "",
                    "top1_file": "",
                    "top1_page": "",
                    "top1_line": "",
                    "top1_score": "",
                    "top1_type": "",
                    "top1_text": "",
                    "error": str(e),
                }
            )
            continue

        match_rank: Optional[int] = None
        matched_summary: Dict[str, object] = {}
        for rank, doc in enumerate(results, start=1):
            if _doc_matches_expected(doc, str(case["expected_file"]), case["expected_page"], case["expected_line"]):
                match_rank = rank
                matched_summary = _doc_result_summary(doc)
                break

        top1_summary = _doc_result_summary(results[0]) if results else {}
        detected_summary = matched_summary if match_rank is not None else top1_summary
        top1_hit = match_rank == 1
        top3_hit = match_rank is not None and match_rank <= top3_window
        if match_rank is not None:
            matched_total += 1
        if top1_hit:
            top1_hits += 1
        if top3_hit:
            top3_hits += 1

        rows.append(
            {
                "query": query_text,
                "expected_file": case["expected_file"],
                "expected_page": case["expected_page"] or "",
                "expected_line": case["expected_line"] or "",
                "notes": case["notes"],
                "row_no": case["row_no"],
                "status": "hit" if match_rank is not None else "miss",
                "first_match_rank": match_rank or "",
                "top1_hit": "Y" if top1_hit else "",
                "top3_hit": "Y" if top3_hit else "",
                "detected_file": detected_summary.get("file", ""),
                "detected_page": detected_summary.get("page", ""),
                "detected_line": detected_summary.get("line", ""),
                "detected_score": detected_summary.get("score", ""),
                "detected_type": detected_summary.get("match_type", ""),
                "detected_text": detected_summary.get("text", ""),
                "top1_file": top1_summary.get("file", ""),
                "top1_page": top1_summary.get("page", ""),
                "top1_line": top1_summary.get("line", ""),
                "top1_score": top1_summary.get("score", ""),
                "top1_type": top1_summary.get("match_type", ""),
                "top1_text": top1_summary.get("text", ""),
                "error": "",
            }
        )

    return {
        "cases": cases,
        "rows": rows,
        "parse_errors": parse_errors,
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "matched_total": matched_total,
        "miss_total": max(total - matched_total, 0),
        "execution_errors": execution_errors,
        "top3_window": top3_window,
    }


def main() -> None:
    st.set_page_config(page_title="Manual PDF RAG", layout="wide")
    st.title("Manual PDF RAG (Chroma Persistent)")
    st.caption("PDFをドラッグ&ドロップ、またはフォルダ指定で一括インデックスできます。")
    st.sidebar.header("対象言語")
    language_options = ["選択してください"] + [label for label, _ in LANGUAGE_OPTIONS]
    active_language_label = st.sidebar.selectbox(
        "コーパスを選択",
        options=language_options,
        index=0,
        key="active_language_label",
    )
    admin_mode = st.sidebar.checkbox("管理者モード", value=False, key="admin_mode")
    st.sidebar.caption("Off: 検索のみ / On: 取り込み・検索・管理")

    if active_language_label == "選択してください":
        st.info("サイドバーで対象言語を選択するとメインメニューを表示します。")
        return

    active_language_code = LANGUAGE_LABEL_TO_CODE[active_language_label]
    st.sidebar.caption("選択した言語ごとに、取り込み・検索・管理・登録ファイル一覧を分けて扱います。")

    app.ensure_dirs()
    if "runtime_data_root_dir" not in st.session_state or "runtime_chroma_root_dir" not in st.session_state:
        loaded_data, loaded_chroma = _load_path_settings()
        st.session_state["runtime_data_root_dir"] = st.session_state.get("runtime_data_dir", loaded_data)
        st.session_state["runtime_chroma_root_dir"] = st.session_state.get("runtime_chroma_dir", loaded_chroma)
    _apply_runtime_paths(
        st.session_state["runtime_data_root_dir"],
        st.session_state["runtime_chroma_root_dir"],
        active_language_code,
    )

    sidebar_restore_result_key = f"sidebar_restore_result_{active_language_code}"
    sidebar_restore_nonce_key = f"sidebar_restore_nonce_{active_language_code}"
    if sidebar_restore_nonce_key not in st.session_state:
        st.session_state[sidebar_restore_nonce_key] = 0
    restore_feedback = st.session_state.pop(sidebar_restore_result_key, None)
    st.sidebar.divider()
    st.sidebar.subheader("検索データ読込")
    st.sidebar.caption("ローカル保存したバックアップzipを復元してから検索します。")
    if restore_feedback:
        if restore_feedback["ok"]:
            st.sidebar.success(restore_feedback["msg"])
        else:
            st.sidebar.error(restore_feedback["msg"])
    sidebar_backup_zip = st.sidebar.file_uploader(
        "バックアップzip",
        type=["zip"],
        key=f"sidebar_backup_zip_{active_language_code}_{st.session_state[sidebar_restore_nonce_key]}",
        help="管理タブで保存したローカルバックアップzipを読み込みます。",
    )
    if st.sidebar.button("バックアップzipを復元", use_container_width=True, key=f"restore_backup_{active_language_code}"):
        if not sidebar_backup_zip:
            st.sidebar.warning("バックアップzipを選択してください。")
        else:
            with st.spinner("ローカルバックアップを復元中..."):
                ok, msg = restore_index_from_local_backup(sidebar_backup_zip)
            st.session_state[sidebar_restore_result_key] = {"ok": ok, "msg": msg}
            if ok:
                st.session_state[sidebar_restore_nonce_key] += 1
                st.session_state.pop("last_upload_signature", None)
            st.rerun()

    if admin_mode:
        debug_result_key = f"debug_chunk_search_result_{active_language_code}"
        evaluation_result_key = f"evaluation_result_{active_language_code}"
        st.sidebar.divider()
        with st.sidebar.expander("登録ファイル確認", expanded=False):
            st.caption("manifest件数とChroma実チャンク数をファイル単位で確認します。")
            file_stats = get_indexed_file_chunk_stats()
            st.caption(
                "manifest登録ファイル数: {manifest_count} / Chroma実ファイル数: {chroma_count} / "
                "Chroma実チャンク総数: {chunk_total}".format(
                    manifest_count=file_stats["manifest_file_count"],
                    chroma_count=file_stats["chroma_file_count"],
                    chunk_total=file_stats["chroma_chunk_total"],
                )
            )
            st.caption(
                "OKファイル: {ok_files} / 要確認ファイル: {problem_files}".format(
                    ok_files=file_stats["ok_files"],
                    problem_files=file_stats["problem_files"],
                )
            )
            if file_stats["rows"]:
                st.dataframe(file_stats["rows"], use_container_width=True, hide_index=True)
            else:
                st.info("現在のコーパスには確認対象ファイルがありません。")

        with st.sidebar.expander("デバッグ検索", expanded=False):
            st.caption("保存済みチャンク全文に対して、生文字列一致と空白正規化一致を確認します。")
            debug_query = st.text_area(
                "確認したい文言",
                height=120,
                key=f"debug_query_{active_language_code}",
                placeholder="例: An attempt to release a wafer is made ...",
            )
            debug_limit = st.number_input(
                "表示上限",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key=f"debug_limit_{active_language_code}",
            )
            if st.button("保存済みチャンクを確認", use_container_width=True, key=f"debug_search_button_{active_language_code}"):
                if not debug_query.strip():
                    st.sidebar.warning("確認したい文言を入力してください。")
                else:
                    result = debug_search_stored_chunks(debug_query.strip(), max_hits=int(debug_limit))
                    st.session_state[debug_result_key] = result
            debug_result = st.session_state.get(debug_result_key)
            if debug_result:
                st.caption(f"走査チャンク数: {debug_result['total_chunks']}")
                raw_hits = debug_result["raw_hits"]
                normalized_hits = debug_result["normalized_hits"]
                st.caption(f"生文字列一致: {len(raw_hits)} 件")
                for idx, hit in enumerate(raw_hits, start=1):
                    page_text = hit["page"] if hit["page"] is not None else "(不明)"
                    st.markdown(
                        f"{idx}. `{hit['file_name']}` / p.{page_text} / 行:{hit['start_line']}-{hit['end_line']}"
                    )
                    st.code(str(hit["snippet"] or "(スニペットなし)"), language="text")
                st.caption(f"空白正規化一致: {len(normalized_hits)} 件")
                for idx, hit in enumerate(normalized_hits, start=1):
                    page_text = hit["page"] if hit["page"] is not None else "(不明)"
                    st.markdown(
                        f"{idx}. `{hit['file_name']}` / p.{page_text} / 行:{hit['start_line']}-{hit['end_line']}"
                    )
                    st.code(str(hit["snippet"] or "(スニペットなし)"), language="text")

        st.sidebar.divider()
        with st.sidebar.expander("評価", expanded=False):
            st.caption("現在の検索ロジックをそのまま使って、正解付きCSVを一括評価します。")
            st.caption("必須列: query, expected_file / 推奨列: expected_page, expected_line, notes")
            st.code(
                "query,expected_file,expected_page,expected_line,notes\n"
                "\"An attempt to release a wafer is made, but the wafer cannot be released normally\","
                "streamlit_uploads/90202-1134DJ.pdf,12,34,example",
                language="text",
            )
            evaluation_csv = st.file_uploader(
                "評価CSV",
                type=["csv"],
                key=f"evaluation_csv_{active_language_code}",
                help="UTF-8 / UTF-8(BOM) / CP932 の CSV を読み込みます。",
            )
            evaluation_top_k = st.number_input(
                "評価対象の上位件数",
                min_value=1,
                max_value=30,
                value=min(5, max(1, int(app.TOP_K))),
                step=1,
                key=f"evaluation_top_k_{active_language_code}",
            )

            if st.button("CSVを一括評価", use_container_width=True, key=f"evaluation_run_{active_language_code}"):
                if not evaluation_csv:
                    st.sidebar.warning("評価CSVを選択してください。")
                else:
                    progress = st.progress(0, text="評価を開始します...")
                    progress_text = st.empty()

                    def on_eval_progress(done: int, total: int, query: str) -> None:
                        safe_total = max(total, 1)
                        ratio = int(done * 100 / safe_total)
                        progress.progress(min(ratio, 100), text=f"評価中 {ratio}%")
                        progress_text.caption(f"{done}/{total} 件: {query[:60]}")

                    try:
                        result = run_evaluation_cases(
                            evaluation_csv,
                            top_k=int(evaluation_top_k),
                            progress_cb=on_eval_progress,
                        )
                        result["top_k"] = int(evaluation_top_k)
                        result["csv_bytes"] = _build_evaluation_csv(result["rows"])
                        result["file_name"] = _evaluation_file_name(active_language_code)
                        st.session_state[evaluation_result_key] = result
                        progress.progress(100, text="評価完了")
                        progress_text.empty()
                    except Exception as e:
                        st.session_state[evaluation_result_key] = {"error": str(e)}
                        progress.empty()
                        progress_text.empty()

            evaluation_result = st.session_state.get(evaluation_result_key)
            if evaluation_result:
                if evaluation_result.get("error"):
                    st.error(str(evaluation_result["error"]))
                else:
                    total_cases = len(evaluation_result["cases"])
                    top1_hits = int(evaluation_result["top1_hits"])
                    top3_hits = int(evaluation_result["top3_hits"])
                    matched_total = int(evaluation_result["matched_total"])
                    miss_total = int(evaluation_result["miss_total"])
                    parse_errors = list(evaluation_result["parse_errors"])
                    execution_errors = int(evaluation_result["execution_errors"])
                    top3_window = int(evaluation_result["top3_window"])

                    def rate_text(hit_count: int, total_count: int) -> str:
                        if total_count <= 0:
                            return "0/0 (0.0%)"
                        return f"{hit_count}/{total_count} ({(hit_count / total_count) * 100:.1f}%)"

                    if total_cases == 0:
                        st.error("有効な評価行が 0 件でした。CSV の列名、区切り文字（`,` / `;` / `|` / タブ）、数値列を確認してください。")
                    st.caption(f"評価件数: {total_cases} / 評価上位件数: {evaluation_result['top_k']}")
                    st.caption(f"Top1一致率: {rate_text(top1_hits, total_cases)}")
                    st.caption(f"Top{top3_window}一致率: {rate_text(top3_hits, total_cases)}")
                    st.caption(f"検出件数: {rate_text(matched_total, total_cases)}")
                    st.caption(f"未検出件数: {miss_total}")
                    if execution_errors:
                        st.warning(f"検索実行エラー: {execution_errors} 件")
                    if parse_errors:
                        st.warning(f"CSV読込エラー: {len(parse_errors)} 件")
                        with st.expander("CSV読込エラー一覧", expanded=False):
                            for msg in parse_errors:
                                st.write(f"- {msg}")

                    result_rows = evaluation_result["rows"]
                    miss_rows = [row for row in result_rows if row.get("status") == "miss"]
                    error_rows = [row for row in result_rows if row.get("status") == "error"]
                    if miss_rows:
                        with st.expander("未検出プレビュー", expanded=False):
                            for row in miss_rows[:10]:
                                st.markdown(
                                    f"- {row['row_no']}行目: `{row['expected_file']}` / p.{row['expected_page'] or '(任意)'}"
                                )
                                st.caption(str(row["query"])[:120])
                    if error_rows:
                        with st.expander("検索実行エラー一覧", expanded=False):
                            for row in error_rows[:10]:
                                st.markdown(f"- {row['row_no']}行目: {row['error']}")

                    csv_bytes = evaluation_result.get("csv_bytes", b"")
                    if csv_bytes:
                        st.download_button(
                            "評価結果CSVを保存",
                            data=csv_bytes,
                            file_name=str(evaluation_result["file_name"]),
                            mime="text/csv",
                            use_container_width=True,
                            key=f"evaluation_download_{active_language_code}",
                        )

    if admin_mode:
        tab_ingest, tab_query, tab_manage = st.tabs(["取り込み", "検索", "管理"])
    else:
        tab_query = st.tabs(["検索"])[0]

    if admin_mode:
        with tab_ingest:
            upload_result_key = f"upload_index_result_{active_language_code}"
            uploader_nonce_key = f"uploaded_pdf_nonce_{active_language_code}"
            if uploader_nonce_key not in st.session_state:
                st.session_state[uploader_nonce_key] = 0

            st.subheader("1) PDFドラッグ&ドロップ取り込み")
            st.caption(f"現在の対象言語: {active_language_label}")
            prior_upload_result = st.session_state.pop(upload_result_key, None)
            if prior_upload_result:
                st.success(
                    "完了: 更新ファイル={touched}, 追加チャンク={added}, 置換削除チャンク={removed}".format(
                        touched=prior_upload_result["touched"],
                        added=prior_upload_result["added"],
                        removed=prior_upload_result["removed"],
                    )
                )
                st.caption(
                    "登録ファイル数: {before} -> {after} / 追加ファイル={added_files} / 更新ファイル={updated_files} / スキップ={skipped_files}".format(
                        before=prior_upload_result.get("before_file_count", 0),
                        after=prior_upload_result.get("after_file_count", 0),
                        added_files=prior_upload_result.get("added_file_count", 0),
                        updated_files=prior_upload_result.get("updated_file_count", 0),
                        skipped_files=prior_upload_result.get("skipped_file_count", 0),
                    )
                )
                if prior_upload_result.get("skipped_file_count", 0):
                    st.warning("変更なしとしてスキップされたファイルがあります。")
                    with st.expander("スキップファイル一覧", expanded=False):
                        for name in prior_upload_result.get("skipped_files", []):
                            st.write(f"- {name}")
                if prior_upload_result["errors"]:
                    st.warning(f"失敗ファイル: {len(prior_upload_result['errors'])} 件")
                    with st.expander("失敗ファイル一覧", expanded=False):
                        for msg in prior_upload_result["errors"]:
                            st.write(f"- {msg}")
            auto_upload_index = st.checkbox(
                "アップロード後に自動で取り込み開始",
                value=True,
                help="ONの場合、ドラッグ&ドロップ直後にインデックス処理を開始します。",
            )
            pdf_password_input = st.text_input(
                "PDFパスワード（必要な場合のみ）",
                type="password",
                key=f"pdf_password_input_{active_language_code}",
                help="パスワード付きPDFを取り込むときだけ入力します。同じパスワードのPDFをまとめて処理できます。",
            )
            pdf_password_candidates = _parse_pdf_password_candidates(pdf_password_input)
            st.caption("異なるパスワードが混在する場合は、パスワードごとに分けて取り込んでください。")
            uploaded_files = st.file_uploader(
                "PDFを複数選択またはドラッグ&ドロップ",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"uploaded_pdf_files_{active_language_code}_{st.session_state[uploader_nonce_key]}",
            )
            if uploaded_files:
                st.info(f"アップロード済み: {len(uploaded_files)} 件")
                total_size_mb = sum(_uploaded_size(f) for f in uploaded_files) / (1024 * 1024)
                st.caption(f"合計サイズ: {total_size_mb:.1f} MB")
                with st.expander("アップロードファイル一覧", expanded=False):
                    for f in uploaded_files:
                        st.write(f"- {f.name}")
            else:
                st.session_state.pop("last_upload_signature", None)
                st.caption("PDFをドロップ後、ボタン押下または自動取り込みで開始します。")

            def run_uploaded_indexing() -> None:
                progress = st.progress(0, text="取り込みを開始します...")
                status = st.empty()
                stage_log = st.empty()
                st.caption("初回は埋め込みモデルのロードで時間がかかる場合があります。")

                def on_upload_progress(ratio01: float, message: str, t: int, a: int, r: int) -> None:
                    ratio = int(ratio01 * 100)
                    progress.progress(
                        ratio,
                        text=f"処理中 {ratio}%: {message}",
                    )
                    status.info(f"更新ファイル={t} / 追加チャンク={a} / 置換削除チャンク={r}")
                    stage_log.caption(f"現在の工程: {message}")

                try:
                    touched, added, removed, errors, summary = index_uploaded_pdfs(
                        uploaded_files,
                        progress_cb=on_upload_progress,
                        pdf_password_candidates=pdf_password_candidates,
                    )
                except Exception as e:
                    status.empty()
                    st.error(f"アップロード取り込みに失敗しました: {e}")
                    return

                completion_text = "取り込み完了" if not errors else "取り込み完了（一部エラーあり）"
                progress.progress(100, text=completion_text)
                status.empty()
                st.session_state[upload_result_key] = {
                    "touched": touched,
                    "added": added,
                    "removed": removed,
                    "errors": errors,
                    "before_file_count": summary.get("before_file_count", 0),
                    "after_file_count": summary.get("after_file_count", 0),
                    "added_file_count": len(summary.get("added_files", [])),
                    "updated_file_count": len(summary.get("updated_files", [])),
                    "skipped_file_count": len(summary.get("skipped_files", [])),
                    "skipped_files": summary.get("skipped_files", [])[:20],
                }
                st.session_state[uploader_nonce_key] += 1
                st.session_state.pop("last_upload_signature", None)
                st.rerun()

            if st.button("アップロードPDFをインデックス", use_container_width=True):
                if not uploaded_files:
                    st.warning("PDFが選択されていません。")
                else:
                    run_uploaded_indexing()

            if auto_upload_index and uploaded_files:
                if len(uploaded_files) > 1:
                    st.info(
                        "複数ファイル一括アップロード時は、自動取り込みを止めています。"
                        " Streamlit Cloud では再実行が重なりやすいため、"
                        " アップロード完了後に手動ボタンで開始してください。"
                    )
                else:
                    sig = _uploaded_signature(uploaded_files)
                    if st.session_state.get("last_upload_signature") != sig:
                        st.session_state["last_upload_signature"] = sig
                        st.toast("アップロードを検出。自動取り込みを開始します。")
                        run_uploaded_indexing()
                    else:
                        st.caption("同一ファイル構成のため自動再実行はスキップ中です。必要なら手動ボタンを押してください。")

            st.divider()
            st.subheader("2) フォルダ指定でPDF一括取り込み")
            folder_input = st.text_input(
                "フォルダパス（再帰的に *.pdf を検索）",
                value=str(app.DATA_DIR.resolve()),
            )
            if st.button("フォルダ内PDFをインデックス", use_container_width=True):
                try:
                    progress = st.progress(0, text="フォルダ走査と取り込みを開始します...")
                    status = st.empty()
                    stage_log = st.empty()

                    def on_folder_progress(ratio01: float, message: str, t: int, a: int, r: int) -> None:
                        ratio = int(ratio01 * 100)
                        progress.progress(
                            ratio,
                            text=f"処理中 {ratio}%: {message}",
                        )
                        status.info(f"更新ファイル={t} / 追加チャンク={a} / 置換削除チャンク={r}")
                        stage_log.caption(f"現在の工程: {message}")

                    scanned, touched, added, removed = index_pdf_folder(
                        folder_input,
                        progress_cb=on_folder_progress,
                        pdf_password_candidates=pdf_password_candidates,
                    )
                    progress.progress(100, text="取り込み完了")
                    status.empty()
                    st.success(
                        f"走査PDF={scanned}, 更新ファイル={touched}, 追加チャンク={added}, 置換削除チャンク={removed}"
                    )
                except Exception as e:
                    st.error(str(e))

    with tab_query:
        st.subheader("類似箇所検索")
        st.caption(f"現在の対象言語: {active_language_label}")
        indexed_names = get_indexed_file_names()
        st.caption(f"現在の登録ファイル数: {len(indexed_names)}")
        if indexed_names:
            preview_names = indexed_names[:5]
            st.caption("登録例: " + " / ".join(preview_names))
            if len(indexed_names) > len(preview_names):
                st.caption(f"ほか {len(indexed_names) - len(preview_names)} 件")
            with st.expander("読み込み済みファイル一覧", expanded=False):
                st.caption("現在の対象言語コーパスに登録されているファイル一覧です。")
                st.code("\n".join(indexed_names), language="text")
        else:
            st.warning(
                "現在のインデックスに登録ファイルがありません。"
                " サイドバーの『バックアップzipを復元』、または管理者モードでの取り込み後に検索してください。"
            )
        search_text = st.text_area(
            "検索文",
            height=140,
            placeholder="例: 探したい文・単語・段落をそのまま入力",
        )
        k = st.number_input("表示件数", min_value=1, max_value=30, value=app.TOP_K, step=1)
        with st.expander("ランキング設定", expanded=False):
            min_score = st.slider("最終スコアの下限", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            min_vector_score = st.slider("ベクトル類似度の下限", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            min_lexical_score = st.slider("語彙一致率の下限", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            candidate_multiplier = st.slider("ベクトル候補倍率", min_value=1, max_value=10, value=6, step=1)
        st.caption("厳密一致・空白差異を無視した一致を優先し、その後に類似度順で並べます。")
        st.caption("行はPDF抽出テキストのページ内行番号です（見た目の行とは一致しない場合があります）。")
        if st.button("類似箇所を検索", use_container_width=True):
            if not search_text.strip():
                st.warning("検索文を入力してください。")
            else:
                docs = run_query(
                    search_text.strip(),
                    int(k),
                    min_score=float(min_score),
                    min_vector_score=float(min_vector_score),
                    min_lexical_score=float(min_lexical_score),
                    candidate_multiplier=int(candidate_multiplier),
                )
                st.markdown(f"### 検索結果（表示: {len(docs)}件）")
                render_sources(docs)

    if admin_mode:
        with tab_manage:
            st.subheader("インデックス管理")
            local_backup_bytes_key = f"local_backup_bytes_{active_language_code}"
            local_backup_name_key = f"local_backup_name_{active_language_code}"
            local_restore_result_key = f"local_restore_result_{active_language_code}"
            local_restore_nonce_key = f"local_restore_nonce_{active_language_code}"
            if local_restore_nonce_key not in st.session_state:
                st.session_state[local_restore_nonce_key] = 0
            if "manage_data_root_input" not in st.session_state:
                st.session_state["manage_data_root_input"] = st.session_state["runtime_data_root_dir"]
            if "manage_chroma_root_input" not in st.session_state:
                st.session_state["manage_chroma_root_input"] = st.session_state["runtime_chroma_root_dir"]

            with st.expander("保存先Path設定", expanded=False):
                data_col1, data_col2 = st.columns([6, 1])
                data_col1.text_input(
                    "DataルートフォルダPath",
                    key="manage_data_root_input",
                    help="言語別データを管理するベースフォルダ。英語は配下の en/ を使用します。",
                )
                if data_col2.button("参照", key="browse_data_dir"):
                    selected = _pick_folder_dialog(st.session_state["manage_data_root_input"])
                    if selected:
                        st.session_state["manage_data_root_input"] = selected
                        st.rerun()
                    if os.name != "nt":
                        st.warning("フォルダ選択ダイアログはWindowsのみ対応です。")

                chroma_col1, chroma_col2 = st.columns([6, 1])
                chroma_col1.text_input(
                    "ChromaルートフォルダPath",
                    key="manage_chroma_root_input",
                    help="日本語・英語の両インデックスを保持するベースフォルダ",
                )
                if chroma_col2.button("参照", key="browse_chroma_dir"):
                    selected = _pick_folder_dialog(st.session_state["manage_chroma_root_input"])
                    if selected:
                        st.session_state["manage_chroma_root_input"] = selected
                        st.rerun()
                    if os.name != "nt":
                        st.warning("フォルダ選択ダイアログはWindowsのみ対応です。")

                if st.button("Pathを適用", use_container_width=True):
                    try:
                        data_path, chroma_path = _apply_runtime_paths(
                            st.session_state["manage_data_root_input"],
                            st.session_state["manage_chroma_root_input"],
                            active_language_code,
                        )
                        st.session_state["runtime_data_root_dir"] = st.session_state["manage_data_root_input"]
                        st.session_state["runtime_chroma_root_dir"] = st.session_state["manage_chroma_root_input"]
                        st.session_state["manage_data_root_input"] = st.session_state["runtime_data_root_dir"]
                        st.session_state["manage_chroma_root_input"] = st.session_state["runtime_chroma_root_dir"]
                        _save_path_settings(
                            st.session_state["runtime_data_root_dir"],
                            st.session_state["runtime_chroma_root_dir"],
                        )
                        st.success(f"適用しました。data={data_path} / chroma={chroma_path}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Path適用に失敗しました: {e}")

            st.caption(f"現在の対象言語: {active_language_label}")
            st.caption(f"現在のdata: {app.DATA_DIR}")
            st.caption(f"現在のchroma(root): {app.CHROMA_DIR}")
            st.caption(f"現在のcollection: {app.COLLECTION_NAME}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("このPathで再インデックス(index)", use_container_width=True):
                    try:
                        with st.spinner("index実行中..."):
                            app.process_files(incremental=True, remove_deleted=True)
                        st.success("indexが完了しました。")
                    except Exception as e:
                        st.error(f"index実行に失敗しました: {e}")
            with col2:
                if st.button("このPathで差分追加(add)", use_container_width=True):
                    try:
                        with st.spinner("add実行中..."):
                            app.process_files(incremental=True, remove_deleted=False)
                        st.success("addが完了しました。")
                    except Exception as e:
                        st.error(f"add実行に失敗しました: {e}")

            st.divider()
            st.subheader("ローカルバックアップ")
            st.caption("サーバー上の検索データを zip で保存し、必要時にローカルから復元します。GitHub には保存しません。")
            if restore_feedback:
                st.info("サイドバーから復元した結果を反映済みです。必要ならこのまま検索できます。")
            restore_result = st.session_state.pop(local_restore_result_key, None)
            if restore_result:
                if restore_result["ok"]:
                    st.success(restore_result["msg"])
                else:
                    st.error(restore_result["msg"])
            backup_col1, backup_col2 = st.columns(2)
            with backup_col1:
                if st.button("バックアップzipを準備", use_container_width=True):
                    with st.spinner("ローカルバックアップを準備中..."):
                        zip_data, msg = build_local_backup_zip()
                    if zip_data is None:
                        st.warning(msg)
                        st.session_state.pop(local_backup_bytes_key, None)
                        st.session_state.pop(local_backup_name_key, None)
                    else:
                        st.session_state[local_backup_bytes_key] = zip_data
                        st.session_state[local_backup_name_key] = _backup_file_name()
                        st.success(msg)
                backup_bytes = st.session_state.get(local_backup_bytes_key)
                backup_name = st.session_state.get(local_backup_name_key, _backup_file_name())
                if backup_bytes:
                    st.download_button(
                        "💾 ローカルへバックアップ保存",
                        data=backup_bytes,
                        file_name=backup_name,
                        mime="application/zip",
                        use_container_width=True,
                    )
                else:
                    st.caption("まず『バックアップzipを準備』を押してください。")
            with backup_col2:
                restore_zip = st.file_uploader(
                    "ローカルのバックアップzipをアップロード",
                    type=["zip"],
                    key=f"manage_restore_zip_{active_language_code}_{st.session_state[local_restore_nonce_key]}",
                    help="保存済みのバックアップzipを選択すると、現在の検索データを置き換えて復元します。",
                )
                if st.button("ローカルzipから復元", use_container_width=True):
                    if not restore_zip:
                        st.warning("バックアップzipを選択してください。")
                    else:
                        with st.spinner("ローカルバックアップを復元中..."):
                            ok, msg = restore_index_from_local_backup(restore_zip)
                        st.session_state[local_restore_result_key] = {"ok": ok, "msg": msg}
                        if ok:
                            st.session_state[local_restore_nonce_key] += 1
                            st.session_state.pop("last_upload_signature", None)
                        st.rerun()

            st.divider()
            st.subheader("サーバー上データ")
            st.caption("検索終了後に押すと、サーバー上に残っているインデックスとアップロード一時ファイルを削除します。")
            confirm_clear = st.checkbox(
                "サーバー上データの全削除を確認",
                value=False,
                key=f"confirm_clear_server_data_{active_language_code}",
            )
            if st.button("サーバー上データを全削除", use_container_width=True, disabled=not confirm_clear):
                with st.spinner("サーバー上データを削除中..."):
                    ok, msg = clear_server_runtime_data(
                        st.session_state["runtime_data_root_dir"],
                        st.session_state["runtime_chroma_root_dir"],
                    )
                st.session_state.pop("last_upload_signature", None)
                st.session_state.pop(local_backup_bytes_key, None)
                st.session_state.pop(local_backup_name_key, None)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

            st.divider()
            manifest = app.load_manifest()
            names = sorted(manifest.get("files", {}).keys())
            st.write(f"登録ファイル数: {len(names)}")
            selected = st.selectbox("削除対象ファイル", options=names if names else ["(なし)"])
            if st.button("選択ファイルを削除", use_container_width=True):
                if not names:
                    st.info("削除対象がありません。")
                elif selected == "(なし)":
                    st.info("削除対象がありません。")
                else:
                    removed = delete_indexed_file(selected)
                    st.success(f"削除完了: {selected} ({removed} chunks)")


if __name__ == "__main__":
    main()
