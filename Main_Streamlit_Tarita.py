import hashlib
import json
import base64
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import streamlit as st

import app

PATH_SETTINGS_FILE = Path(__file__).with_name("path_settings.json")


def _get_upload_cache_dir() -> Path:
    return app.DATA_DIR / "streamlit_uploads"


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


def _apply_runtime_paths(data_dir: str, chroma_dir: str) -> Tuple[Path, Path]:
    data_path = Path(data_dir).expanduser().resolve()
    chroma_path = Path(chroma_dir).expanduser().resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    chroma_path.mkdir(parents=True, exist_ok=True)
    app.DATA_DIR = data_path
    app.CHROMA_DIR = chroma_path
    app.MANIFEST_PATH = chroma_path / "index_manifest.json"
    return data_path, chroma_path


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_key(name: str) -> str:
    return name.replace("\\", "/")


def _upsert_pdf(
    logical_file_name: str,
    file_path: Path,
    file_hash: str,
    store,
    manifest_files: Dict,
    stage_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, int, int]:
    logical_file_name = _normalize_key(logical_file_name)
    old = manifest_files.get(logical_file_name)
    changed = (old is None) or (old.get("hash") != file_hash)
    if not changed:
        return False, 0, 0

    removed = 0
    if old and old.get("chunk_ids"):
        if stage_cb:
            stage_cb("既存チャンクを削除中...")
        app.delete_by_ids(store, old["chunk_ids"])
        removed = len(old["chunk_ids"])

    if stage_cb:
        stage_cb("PDFテキスト抽出中...")
    content = app.read_pdf(file_path)
    if stage_cb:
        stage_cb("チャンク分割中...")
    docs = app.split_with_metadata(
        file_name=logical_file_name,
        content=content,
        chunk_size=app.CHUNK_SIZE,
        overlap=app.CHUNK_OVERLAP,
        source_path=str(file_path.resolve()),
    )
    ids = [d.metadata["chunk_id"] for d in docs]

    if docs:
        if stage_cb:
            stage_cb("ベクトル化してChromaへ保存中...")
        store.add_documents(docs, ids=ids)

    manifest_files[logical_file_name] = {
        "hash": file_hash,
        "chunk_ids": ids,
        "updated_at": _utc_now_iso(),
    }
    return True, len(ids), removed


def index_uploaded_pdfs(
    uploaded_files,
    progress_cb: Optional[Callable[[float, str, int, int, int], None]] = None,
) -> Tuple[int, int, int]:
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

    touched = 0
    added = 0
    removed = 0

    for idx, uploaded in enumerate(uploaded_files, start=1):
        file_start = 0.15 + 0.8 * ((idx - 1) / total)
        file_span = 0.8 / total
        emit(file_start, f"{uploaded.name}: 取り込み準備中...", touched, added, removed)
        data = uploaded.getvalue()
        if not data:
            emit(file_start + file_span, f"{uploaded.name}: 空ファイルのためスキップ", touched, added, removed)
            continue
        save_path = upload_cache_dir / uploaded.name
        save_path.write_bytes(data)

        logical_name = f"streamlit_uploads/{uploaded.name}"
        stage_ratio = {
            "既存チャンクを削除中...": 0.20,
            "PDFテキスト抽出中...": 0.45,
            "チャンク分割中...": 0.70,
            "ベクトル化してChromaへ保存中...": 0.90,
        }

        def on_stage(msg: str, name: str = uploaded.name) -> None:
            ratio = stage_ratio.get(msg, 0.5)
            emit(file_start + file_span * ratio, f"{name}: {msg}", touched, added, removed)

        changed, a, r = _upsert_pdf(
            logical_file_name=logical_name,
            file_path=save_path,
            file_hash=_sha256_bytes(data),
            store=store,
            manifest_files=manifest_files,
            stage_cb=on_stage if progress_cb else None,
        )
        if changed:
            touched += 1
            added += a
            removed += r
        emit(file_start + file_span, f"{uploaded.name}: 完了", touched, added, removed)

    emit(0.97, "結果保存中...", touched, added, removed)
    app.save_manifest(manifest)
    emit(1.0, "取り込み完了", touched, added, removed)
    return touched, added, removed


def _uploaded_signature(uploaded_files) -> str:
    if not uploaded_files:
        return ""
    parts = []
    for f in uploaded_files:
        size = len(f.getvalue() or b"")
        parts.append(f"{f.name}:{size}")
    return "|".join(sorted(parts))


def index_pdf_folder(
    folder_path: str,
    progress_cb: Optional[Callable[[float, str, int, int, int], None]] = None,
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

    pdf_files = sorted(p for p in root.rglob("*.pdf") if p.is_file())
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
        )
        if changed:
            touched += 1
            added += a
            removed += r
        emit(file_start + file_span, f"{rel}: 完了", touched, added, removed)

    emit(0.97, "結果保存中...", touched, added, removed)
    app.save_manifest(manifest)
    emit(1.0, "取り込み完了", touched, added, removed)
    return len(pdf_files), touched, added, removed


def run_query(
    question: str,
    k: int,
    relevance_threshold: float,
    min_vector_score: float,
    min_lexical_score: float,
    candidate_multiplier: int,
) -> Tuple[str, List]:
    docs = app.search_relevant_docs(
        question,
        k=k,
        relevance_threshold=relevance_threshold,
        min_vector_score=min_vector_score,
        min_lexical_score=min_lexical_score,
        candidate_multiplier=candidate_multiplier,
    )
    if not docs:
        return "該当情報が見つかりませんでした。", []
    answer = app.llm_answer(question, docs)
    return answer, docs


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
        st.write("(なし)")
        return
    for idx, d in enumerate(docs):
        md = d.metadata
        file_name = md.get("file_name", "unknown")
        start_line = md.get("start_line", "?")
        end_line = md.get("end_line", "?")
        page = md.get("page_start", md.get("page"))
        page_end = md.get("page_end", page)
        source_path = md.get("source_path")
        if page is not None and page_end is not None and page_end != page:
            page_text = f"{page}-{page_end}"
        elif page is not None:
            page_text = str(page)
        else:
            page_text = "(不明)"

        expander_title = f"{idx + 1}. {file_name} / ページ:{page_text} / 行:{start_line}-{end_line}"
        with st.expander(expander_title, expanded=(idx == 0)):
            st.markdown(f"- ファイル: `{file_name}`")
            st.markdown(f"- ページ: `{page_text}`")
            st.markdown(f"- 行(ページ内): `{start_line}-{end_line}`")
            score = md.get("relevance_score")
            if score is not None:
                st.markdown(f"- 関連度スコア: `{score}`")

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

            st.markdown("**該当箇所プレビュー（抽出チャンク）**")
            st.code(d.page_content.strip() or "(テキストなし)", language="text")


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


def main() -> None:
    st.set_page_config(page_title="Manual PDF RAG", layout="wide")
    st.title("Manual PDF RAG (Chroma Persistent)")
    st.caption("PDFをドラッグ&ドロップ、またはフォルダ指定で一括インデックスできます。")

    app.ensure_dirs()
    if "runtime_data_dir" not in st.session_state or "runtime_chroma_dir" not in st.session_state:
        loaded_data, loaded_chroma = _load_path_settings()
        st.session_state["runtime_data_dir"] = loaded_data
        st.session_state["runtime_chroma_dir"] = loaded_chroma
    _apply_runtime_paths(
        st.session_state["runtime_data_dir"],
        st.session_state["runtime_chroma_dir"],
    )

    tab_ingest, tab_query, tab_manage = st.tabs(["取り込み", "検索", "管理"])

    with tab_ingest:
        st.subheader("1) PDFドラッグ&ドロップ取り込み")
        auto_upload_index = st.checkbox(
            "アップロード後に自動で取り込み開始",
            value=True,
            help="ONの場合、ドラッグ&ドロップ直後にインデックス処理を開始します。",
        )
        uploaded_files = st.file_uploader(
            "PDFを複数選択またはドラッグ&ドロップ",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.info(f"アップロード済み: {len(uploaded_files)} 件")
            with st.expander("アップロードファイル一覧", expanded=False):
                for f in uploaded_files:
                    st.write(f"- {f.name}")
        else:
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

            touched, added, removed = index_uploaded_pdfs(uploaded_files, progress_cb=on_upload_progress)
            progress.progress(100, text="取り込み完了")
            status.empty()
            st.success(
                f"完了: 更新ファイル={touched}, 追加チャンク={added}, 置換削除チャンク={removed}"
            )

        if st.button("アップロードPDFをインデックス", use_container_width=True):
            if not uploaded_files:
                st.warning("PDFが選択されていません。")
            else:
                run_uploaded_indexing()

        if auto_upload_index and uploaded_files:
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
                )
                progress.progress(100, text="取り込み完了")
                status.empty()
                st.success(
                    f"走査PDF={scanned}, 更新ファイル={touched}, 追加チャンク={added}, 置換削除チャンク={removed}"
                )
            except Exception as e:
                st.error(str(e))

    with tab_query:
        st.subheader("RAG検索")
        question = st.text_area("質問", height=120, placeholder="例: インストール手順の前提条件は？")
        k = st.number_input("Top-k", min_value=1, max_value=20, value=app.TOP_K, step=1)
        with st.expander("関連度フィルタ設定（ノイズ低減）", expanded=False):
            relevance_threshold = st.slider("最終関連度しきい値", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
            min_vector_score = st.slider("ベクトル類似度の下限", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
            min_lexical_score = st.slider("語彙一致率の下限", min_value=0.0, max_value=1.0, value=0.03, step=0.01)
            candidate_multiplier = st.slider("再評価候補倍率", min_value=1, max_value=10, value=6, step=1)
        st.caption("行はPDF抽出テキストのページ内行番号です（見た目の行とは一致しない場合があります）。")
        if st.button("検索して回答", use_container_width=True):
            if not question.strip():
                st.warning("質問を入力してください。")
            else:
                answer, docs = run_query(
                    question.strip(),
                    int(k),
                    relevance_threshold=float(relevance_threshold),
                    min_vector_score=float(min_vector_score),
                    min_lexical_score=float(min_lexical_score),
                    candidate_multiplier=int(candidate_multiplier),
                )
                st.markdown("### 回答")
                st.write(answer)
                st.markdown(f"### 出典（表示: {len(docs)}件）")
                render_sources(docs)

    with tab_manage:
        st.subheader("インデックス管理")
        if "manage_data_dir_input" not in st.session_state:
            st.session_state["manage_data_dir_input"] = st.session_state["runtime_data_dir"]
        if "manage_chroma_dir_input" not in st.session_state:
            st.session_state["manage_chroma_dir_input"] = st.session_state["runtime_chroma_dir"]

        with st.expander("保存先Path設定", expanded=False):
            data_col1, data_col2 = st.columns([6, 1])
            data_col1.text_input(
                "DataフォルダPath",
                key="manage_data_dir_input",
                help="取り込み対象PDFやアップロード保存先を管理するフォルダ",
            )
            if data_col2.button("参照", key="browse_data_dir"):
                selected = _pick_folder_dialog(st.session_state["manage_data_dir_input"])
                if selected:
                    st.session_state["manage_data_dir_input"] = selected
                    st.rerun()
                if os.name != "nt":
                    st.warning("フォルダ選択ダイアログはWindowsのみ対応です。")

            chroma_col1, chroma_col2 = st.columns([6, 1])
            chroma_col1.text_input(
                "ChromaフォルダPath",
                key="manage_chroma_dir_input",
                help="ベクトルDBとmanifestを保持するフォルダ",
            )
            if chroma_col2.button("参照", key="browse_chroma_dir"):
                selected = _pick_folder_dialog(st.session_state["manage_chroma_dir_input"])
                if selected:
                    st.session_state["manage_chroma_dir_input"] = selected
                    st.rerun()
                if os.name != "nt":
                    st.warning("フォルダ選択ダイアログはWindowsのみ対応です。")

            if st.button("Pathを適用", use_container_width=True):
                try:
                    data_path, chroma_path = _apply_runtime_paths(
                        st.session_state["manage_data_dir_input"],
                        st.session_state["manage_chroma_dir_input"],
                    )
                    st.session_state["runtime_data_dir"] = str(data_path)
                    st.session_state["runtime_chroma_dir"] = str(chroma_path)
                    st.session_state["manage_data_dir_input"] = str(data_path)
                    st.session_state["manage_chroma_dir_input"] = str(chroma_path)
                    _save_path_settings(str(data_path), str(chroma_path))
                    st.success(f"適用しました。data={data_path} / chroma={chroma_path}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Path適用に失敗しました: {e}")

        st.caption(f"現在のdata: {app.DATA_DIR}")
        st.caption(f"現在のchroma: {app.CHROMA_DIR}")
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
