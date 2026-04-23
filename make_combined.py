# make_combined.py
# 从 wiki/faq/*.md 生成 Dify 友好的合并文件
# 每个 FAQ 是一个纯文本块，FAQ 之间用 \n\n 分隔
# 块内无 \n\n，Dify 按 \n\n 精准切成 167 个 chunk

import re
from pathlib import Path


def strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5:]
    return text


def to_plain(text: str) -> str:
    # 统一行尾
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 把 ## 变体问法 / ## 参见 改成普通标签，避免 Dify markdown 分段
    text = re.sub(r"\n## 变体问法\n", "\n变体问法：", text)
    text = re.sub(r"\n## 参见\n", "\n参见：", text)
    # 压掉所有剩余连续空行
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


faq_dir = Path("wiki/faq")
out     = Path("raw/combined_faq.md")

parts = []
for md in sorted(faq_dir.glob("*.md")):
    content = md.read_text(encoding="utf-8")
    content = strip_frontmatter(content)
    content = to_plain(content)
    if content:
        parts.append(content)

# 用二进制写出，保证纯 \n 行尾（不被 Windows 转成 \r\n）
out.write_bytes(("\n\n".join(parts)).encode("utf-8"))
print(f"写入 {len(parts)} 个 FAQ → {out}")
