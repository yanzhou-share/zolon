"""
文本清洗模块

提供文本清洗功能：移除特殊字符、合并空白、去除重复、过滤页眉页脚等。
"""

import re


def clean_text(text: str) -> str:
    """完整的文本清洗流程"""
    if not text:
        return ""

    text = _remove_control_chars(text)
    print(f"【移除控制字符】{text}")
    text = _normalize_whitespace(text)
    print(f"【合并多余空白】{text}")
    text = _remove_repeated_lines(text)
    print(f"【移除重复行】{text}")
    text = _remove_header_footer(text)
    print(f"【移除页眉页脚】{text}")
    text = _filter_toc_content(text)    
    print(f"【过滤目录内容】{text}")
    text = _remove_consecutive_duplicates(text)
    print(f"【移除连续重复】{text}")
    text = text.strip()
    print(f"【移首尾空格】{text}")

    return text


def _remove_control_chars(text: str) -> str:
    """移除不可打印字符和控制字符，保留中文、英文、数字、标点"""
    # 保留：中文、英文、数字、常见标点、换行、空格
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # 移除零宽字符
    cleaned = re.sub(r'[\u200b-\u200f\u2028-\u202f\u2060-\u2069\ufeff]', '', cleaned)
    return cleaned


def _normalize_whitespace(text: str) -> str:
    """合并多余空白"""
    # 合并连续空格（保留换行）
    text = re.sub(r'[^\S\n]+', ' ', text)
    # 合并连续空行（最多保留2个）
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去除每行首尾空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text


def _remove_repeated_lines(text: str) -> str:
    """移除连续重复的行"""
    lines = text.split('\n')
    result = []
    prev_line = ""

    for line in lines:
        stripped = line.strip()
        if stripped and stripped == prev_line:
            continue
        result.append(line)
        prev_line = stripped

    return '\n'.join(result)


def _remove_header_footer(text: str) -> str:
    """移除页眉页脚（检测文档首尾重复内容）"""
    lines = text.split('\n')
    if len(lines) < 10:
        return text

    # 检测前3行是否在后3行中重复出现
    header_candidates = lines[:3]
    footer_candidates = lines[-3:]

    # 检查页眉
    for i, line in enumerate(lines[3:], start=3):
        if line.strip() and all(h in line for h in header_candidates if h.strip()):
            lines = lines[i:]
            break

    # 检查页脚：只在 footer_candidates 之前的行中查找重复
    for i in range(len(lines) - 4, max(len(lines) - 9, 0), -1):
        line = lines[i].strip()
        if line and any(f in line for f in footer_candidates if f.strip()):
            lines = lines[:i]
            break

    return '\n'.join(lines)


def _filter_toc_content(text: str) -> str:
    """过滤目录、索引等非正文内容"""
    lines = text.split('\n')
    result = []
    skip_mode = False

    for line in lines:
        stripped = line.strip()

        # 检测目录标题
        if re.match(r'^(目录|目 录|CONTENTS?|TABLE OF CONTENTS?)\s*$', stripped, re.IGNORECASE):
            skip_mode = True
            continue

        # 检测目录条目（页码模式）
        if skip_mode and re.search(r'\.{2,}\s*\d+\s*$', stripped):
            continue

        # 检测独立的页码行
        if re.match(r'^\s*\d+\s*$', stripped):
            continue

        # 检测"第X页"模式
        if re.match(r'^第\s*\d+\s*页', stripped):
            continue

        skip_mode = False
        result.append(line)

    return '\n'.join(result)


def _remove_consecutive_duplicates(text: str) -> str:
    """移除段落级别的重复内容"""
    paragraphs = text.split('\n\n')
    result = []
    seen = set()

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            result.append(para)
            continue

        # 使用前80字符作为去重key
        key = stripped[:80].lower()
        if key not in seen:
            seen.add(key)
            result.append(para)

    return '\n\n'.join(result)


def clean_docx_paragraphs(paragraphs: list) -> list:
    """清洗 docx 提取的段落列表"""
    cleaned = []
    for para in paragraphs:
        if not para or not para.strip():
            continue

        para = para.strip()
        para = _remove_control_chars(para)
        para = re.sub(r'\s+', ' ', para)  # 合并空白

        if para and len(para) > 1:  # 过滤单字符
            cleaned.append(para)

    return cleaned


def clean_table_content(text: str) -> str:
    """清洗表格内容"""
    # 移除表格中的多余换行
    text = re.sub(r'\n\s*\n', '\n', text)
    # 合并单元格内的空白
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
