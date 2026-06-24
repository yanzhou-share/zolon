"""
文档解析模块

负责文档解析、分块、表格处理等功能。
支持 txt/docx/md/pdf 格式。
"""

import os
from config import chunk as chunk_cfg


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list:
    """智能分块：按段落分割，避免重复"""
    chunk_size = chunk_size or chunk_cfg.CHUNK_SIZE
    overlap = overlap or chunk_cfg.CHUNK_OVERLAP

    chunks = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if paragraphs:
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = current_chunk + "\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(para) > chunk_size:
                    for i in range(0, len(para), chunk_size - overlap):
                        chunks.append(para[i:i + chunk_size])
                else:
                    current_chunk = para
        if current_chunk:
            chunks.append(current_chunk)
    else:
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)

    # 去重
    unique_chunks = []
    seen = set()
    for chunk in chunks:
        key = chunk[:100].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    return unique_chunks


def parse_txt(file_path: str) -> str:
    """解析 txt/md 文件"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


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
    return "\n\n".join(text_parts)


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
