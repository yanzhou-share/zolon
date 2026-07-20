"""
文档解析模块

负责文档解析、分块、表格处理等功能。
支持 txt/docx/md/pdf 格式。
"""

import os
from config import chunk as chunk_cfg
from text_cleaner import clean_text, clean_docx_paragraphs


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list:
    """递归分割：按分隔符优先级逐级细化，保证语义完整性"""
    chunk_size = chunk_size or chunk_cfg.CHUNK_SIZE
    overlap = overlap or chunk_cfg.CHUNK_OVERLAP
    separators = ["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    chunks = _recursive_split(text, separators, chunk_size, overlap)
    return chunks


def _recursive_split(text: str, separators: list, chunk_size: int, overlap: int) -> list:
    """按分隔符优先级递归分割文本"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    for sep in separators:
        if sep == "":
            # 最后手段：按字符切
            return [text[i:i + chunk_size].strip()
                    for i in range(0, len(text), chunk_size - overlap)
                    if text[i:i + chunk_size].strip()]

        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        chunks = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size:
                    next_seps = separators[separators.index(sep) + 1:]
                    chunks.extend(_recursive_split(part, next_seps, chunk_size, overlap))
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    return [text]


def chunk_text_semantic(text: str, embed_fn, chunk_size: int = None,
                        similarity_threshold: float = None) -> list:
    """基于语义的智能分块：用 embedding 检测语义断点"""
    import re
    chunk_size = chunk_size or chunk_cfg.CHUNK_SIZE
    threshold = similarity_threshold or chunk_cfg.SEMANTIC_THRESHOLD

    sentences = re.split(r'(?<=[。！？\n])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 3:
        return _recursive_split(text, ["\n\n", "\n", "。", "！", "？", "，", " ", ""], chunk_size, 50)

    embeddings = embed_fn(sentences)

    breaks = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        if sim < threshold:
            breaks.append(i + 1)

    groups = []
    prev = 0
    for brk in breaks:
        groups.append(sentences[prev:brk])
        prev = brk
    groups.append(sentences[prev:])

    chunks = []
    seps = ["。", "！", "？", "，"]
    for group in groups:
        group_text = "".join(group)
        if len(group_text) <= chunk_size:
            if group_text.strip():
                chunks.append(group_text.strip())
        else:
            chunks.extend(_recursive_split(group_text, seps, chunk_size, 50))

    unique_chunks = []
    seen = set()
    for chunk in chunks:
        key = chunk[:100].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    return unique_chunks


def _cosine_similarity(a, b):
    """计算余弦相似度"""
    import numpy as np
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def parse_txt(file_path: str) -> str:
    """解析 txt/md 文件"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return clean_text(text)


def parse_docx(file_path: str) -> str:
    """解析 docx 文件，支持段落、表格和 Q&A 配对"""
    from docx import Document
    from docx.oxml.ns import qn
    doc = Document(file_path)
    parts = []

    paragraphs = []
    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            text = ""
            for child in element.iter():
                if child.text:
                    text += child.text
            if text.strip():
                paragraphs.append(text.strip())
        elif tag == "tbl":
            from docx.table import Table
            table = Table(element, doc)
            table_text = _docx_table_to_markdown(table)
            if table_text:
                paragraphs.append(table_text)

    # 清洗段落
    paragraphs = clean_docx_paragraphs(paragraphs)

    # 合并 Q&A 配对
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        if para.startswith("Q:") or para.startswith("Q："):
            if i + 1 < len(paragraphs):
                next_para = paragraphs[i + 1]
                if next_para.startswith("A:") or next_para.startswith("A："):
                    parts.append(f"{para}\n{next_para}")
                    i += 2
                    continue
        parts.append(para)
        i += 1

    return "\n\n".join(parts)


def _docx_table_to_markdown(table) -> str:
    """将 docx 表格转换为 Markdown 格式"""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append(cells)
    if not rows:
        return ""
    lines = []
    header = rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(lines)


def parse_pdf(file_path: str) -> str:
    """解析 PDF 文件，支持文本和表格提取"""
    import pdfplumber
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            tables = page.extract_tables()
            for table in tables:
                table_text = _pdf_table_to_markdown(table)
                if table_text:
                    text_parts.append(table_text)
    combined = "\n\n".join(text_parts)
    return clean_text(combined)


def _pdf_table_to_markdown(table) -> str:
    """将 PDF 表格转换为 Markdown 格式"""
    if not table:
        return ""
    cleaned = []
    for row in table:
        if row:
            cleaned_row = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
            cleaned.append(cleaned_row)
    if not cleaned:
        return ""
    lines = []
    header = cleaned[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in cleaned[1:]:
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(lines)
