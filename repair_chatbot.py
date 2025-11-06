# -*- coding: utf-8 -*-
"""
Streamlit 앱: 취급 정비지침서 챗봇 (PDF + OCR)
- input_csv 폴더에 PDF 지침서를 넣으면 OCR로 텍스트를 추출해 벡터스토어를 구축하고 질문응답을 제공합니다.
- OCR: PyMuPDF로 페이지 렌더링 → EasyOCR(ko+en) → 본문 텍스트와 병합
- LLM: Google Gemini 1.5 Pro (langchain-google-genai)

필요 패키지(예시)
-----------------
# CPU 환경 기준
pip install streamlit pandas loguru pillow numpy
pip install easyocr pymupdf
pip install langchain langchain-community langchain-google-genai faiss-cpu sentence-transformers

Secrets
-------
[⋯] → Edit secrets 에 GEMINI_API_KEY 를 등록하세요.

폴더 구조
---------
project/
  rechatbot_manual_ocr.py
  input_csv/
    장비정비지침서1.pdf
    장비정비지침서2.pdf

주의
----
- EasyOCR는 최초 실행 시 ko/en 모델을 다운로드합니다(인터넷 필요). 폐쇄망이면 사전 배포가 필요합니다.
- PDF의 텍스트가 추출 가능한 페이지는 OCR을 생략하고, 이미지·도표 중심 페이지는 자동으로 OCR 합니다.
- 대용량 PDF는 시간이 걸릴 수 있으므로, 캐시와 저장 기능을 제공합니다.
"""
from __future__ import annotations
import os
import re
import io
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from loguru import logger

import streamlit as st

# ──────────────────────────────────────────────────────────
# 기본 설정
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="취급 정비지침서 챗봇 (PDF+OCR)",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "input_csv"  # 기존 폴더 재활용
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR = BASE_DIR / ".manual_index"
INDEX_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────

def _clean_text(t: str) -> str:
    t = re.sub(r"[\r\t\f]", " ", t)
    t = re.sub(r"\u200b|\xa0", " ", t)
    t = re.sub(r" +", " ", t).strip()
    return t

@st.cache_resource(show_spinner=False)
def _easyocr_reader():
    import easyocr
    # 한국어/영어 동시 인식
    return easyocr.Reader(['ko', 'en'], gpu=False)  # GPU 환경이면 gpu=True 권장

@st.cache_data(show_spinner=False)
def list_pdf_files() -> List[Path]:
    return sorted([p for p in DATA_DIR.glob('*.pdf') if p.is_file()])

# ──────────────────────────────────────────────────────────
# PDF → 텍스트 (PyMuPDF + EasyOCR)
# ──────────────────────────────────────────────────────────

def _page_to_image_pix(doc, page_index: int, zoom: float = 2.0) -> Image.Image:
    import fitz  # PyMuPDF
    page = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def _page_text_density(text: str, area: Tuple[int, int]) -> float:
    # 간단한 텍스트 밀도 휴리스틱: 글자수 / 면적(백만 px 단위)
    w, h = area
    px_m = max(1, (w * h) / 1_000_000)
    return len(text.strip()) / px_m


def extract_pdf_with_ocr(path: Path, force_ocr: bool = False,
                         density_threshold: float = 800.0) -> List[Dict]:
    """
    각 페이지별로 (page, text, meta) 반환
    - 우선 PyMuPDF의 page.get_text("text")를 사용
    - 텍스트 밀도가 낮거나(force_ocr), 글자 수가 매우 적으면 EasyOCR 실행
    - OCR 텍스트와 추출 텍스트를 병합(중복 제거)
    """
    import fitz  # PyMuPDF

    results: List[Dict] = []
    with fitz.open(path) as doc:
        reader = None
        for i in range(len(doc)):
            page = doc.load_page(i)
            raw = page.get_text("text") or ""
            raw = _clean_text(raw)

            # 텍스트 밀도 판단
            w, h = page.rect.width, page.rect.height
            density = _page_text_density(raw, (int(w), int(h)))

            ocr_text = ""
            if force_ocr or len(raw) < 40 or density < density_threshold:
                # OCR 수행
                if reader is None:
                    reader = _easyocr_reader()
                img = _page_to_image_pix(doc, i, zoom=2.0)
                np_img = np.array(img)
                ocr_result = reader.readtext(np_img, detail=0, paragraph=True)
                ocr_text = _clean_text("\n".join(ocr_result))

            # 병합 (간단 중복 제거)
            merged = raw
            if ocr_text:
                if raw and ocr_text and ocr_text not in raw:
                    merged = (raw + "\n" + ocr_text).strip()
                elif not raw:
                    merged = ocr_text

            results.append({
                "source": str(path.name),
                "page": i + 1,
                "text": merged,
            })
    return results


# ──────────────────────────────────────────────────────────
# 임베딩 & 벡터스토어
# ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _tiktoken_len(text: str) -> int:
    try:
        import tiktoken
        tok = tiktoken.get_encoding("cl100k_base")
        return len(tok.encode(text))
    except Exception:
        return len(text)


def split_to_docs(rows: List[Dict], chunk_size: int = 900, chunk_overlap: int = 120):
    from langchain_core.documents import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_tiktoken_len,
    )

    lang_docs = []
    for r in rows:
        meta = {"source": r["source"], "page": r["page"]}
        for chunk in splitter.split_text(r["text"]):
            lang_docs.append(Document(page_content=chunk, metadata=meta))
    return lang_docs


def build_faiss_index(docs):
    from langchain_community.vectorstores import FAISS
    emb = build_embeddings()
    return FAISS.from_documents(docs, emb)


def save_faiss_index(vs, path: Path):
    vs.save_local(str(path))


def load_faiss_index(path: Path):
    from langchain_community.vectorstores import FAISS
    emb = build_embeddings()
    return FAISS.load_local(str(path), emb, allow_dangerous_deserialization=True)


# ──────────────────────────────────────────────────────────
# LLM 체인 (Gemini 1.5 Pro)
# ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_conversation_chain(vstore, gemini_api_key: str):
    from langchain.chains import ConversationalRetrievalChain
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.memory import ConversationBufferMemory

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=gemini_api_key,
        temperature=0.0,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vstore.as_retriever(search_type="mmr", search_kwargs={"k": 4}),
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        ),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=False,
    )
    return chain


# ──────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────
st.title("🛠️ 취급 정비지침서 챗봇 (PDF + OCR)")
st.caption("PDF 지침서의 텍스트/그림을 OCR로 읽고, Gemini로 문서 Q&A를 제공합니다.")

with st.sidebar:
    st.subheader("🔧 설정")
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        st.info("Secrets에 GEMINI_API_KEY를 등록하세요. (⋯ → Edit secrets)")

    st.markdown("---")
    st.subheader("📄 지침서 스캔 & 인덱스")
    pdfs = list_pdf_files()
    st.write(f"감지된 PDF: **{len(pdfs)}**건")
    st.write("- " + "\n- ".join([p.name for p in pdfs]) if pdfs else "input_csv 폴더에 PDF를 넣으세요.")

    force_ocr = st.checkbox("모든 페이지 OCR 강제", value=False)
    density_th = st.slider("OCR 전환 임계(텍스트 밀도)", 100.0, 2000.0, 800.0, 50.0,
                           help="낮을수록 더 많은 페이지에서 PyMuPDF 텍스트만 사용합니다.")
    do_build = st.button("📚 인덱스 생성/갱신")

# 세션 상태
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 좌측에서 PDF 인덱스를 먼저 생성한 후, 아래 채팅창에 질문해 주세요."}]

# 인덱스 생성
if do_build:
    if not pdfs:
        st.warning("input_csv 폴더에 PDF가 없습니다.")
    elif not gemini_api_key:
        st.warning("Gemini API Key가 필요합니다.")
    else:
        all_rows: List[Dict] = []
        progress = st.progress(0)
        status = st.empty()

        for idx, p in enumerate(pdfs, start=1):
            status.info(f"OCR/추출 중: {p.name} ({idx}/{len(pdfs)})")
            try:
                rows = extract_pdf_with_ocr(p, force_ocr=force_ocr, density_threshold=density_th)
                all_rows.extend(rows)
            except Exception as e:
                st.error(f"{p.name} 처리 실패: {type(e).__name__}: {e}")
            progress.progress(idx / max(1, len(pdfs)))

        status.info("텍스트 분할 및 벡터 인덱스 생성 중...")
        docs = split_to_docs(all_rows)
        vdb = build_faiss_index(docs)

        save_path = INDEX_DIR / "faiss_manual"
        save_faiss_index(vdb, save_path)
        st.session_state.qa_chain = get_conversation_chain(vdb, gemini_api_key)
        status.success("완료! 인덱스 저장 및 QA 체인 준비가 끝났습니다.")
        st.success(f"총 청크 수: {len(docs):,} 개")

# 기존 인덱스 로드 시도
if st.session_state.qa_chain is None:
    saved = INDEX_DIR / "faiss_manual"
    if (saved.with_suffix(".pkl").exists() or (saved / "index.faiss").exists()):
        try:
            vdb = load_faiss_index(saved)
            st.session_state.qa_chain = get_conversation_chain(vdb, st.secrets.get("GEMINI_API_KEY", ""))
            st.info("기존 인덱스를 불러왔습니다.")
        except Exception:
            pass

# 채팅 UI
st.markdown("---")
st.subheader("💬 문서 Q&A")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("정비지침서에 대해 질문하세요 (예: 비상제동 절차, 피견인 운전 절차 등)")
if user_q:
    if st.session_state.qa_chain is None:
        st.warning("먼저 좌측에서 인덱스를 생성하세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("검색 및 응답 생성 중..."):
                result = st.session_state.qa_chain({"question": user_q})
                answer = result.get("answer", "")
                srcs = result.get("source_documents", [])
                st.markdown(answer)
                if srcs:
                    with st.expander("참고 소스"):
                        for i, d in enumerate(srcs[:6], start=1):
                            src = d.metadata.get("source", "unknown")
                            page = d.metadata.get("page", None)
                            meta = f"{src}" + (f" (p.{page})" if isinstance(page, int) else "")
                            st.markdown(f"**{i}.** {meta}")
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.caption(
    "ⓘ 팁: OCR 임계값을 높이면 그림 많은 페이지에서 OCR을 더 자주 수행합니다. 질문은 구체적으로 적을수록 정확도가 높아집니다.")
