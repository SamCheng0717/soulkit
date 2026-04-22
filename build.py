# build.py
#
# 传入聊天记录 CSV，输出知识库 + 人格
# 用法: python build.py <chatlogs.csv>
#
# CSV 必须包含列: session_id, role, message

import os, csv, json, hashlib, sys, argparse
from pathlib import Path
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# ── Schema ────────────────────────────────────────────────────────
def load_schema(base: Path) -> str:
    custom = base / "wiki" / "schema.md"
    if custom.exists():
        return custom.read_text(encoding="utf-8")
    template = Path("templates/schema.md")
    if template.exists():
        return template.read_text(encoding="utf-8")
    return ""

# ── Prompts ───────────────────────────────────────────────────────
FAQ_PROMPT = """{schema}

从以下客服对话提取 FAQ。

判断标准：
- 问题通用（无订单号/姓名/日期） → 可提取
- 客服给出了明确答案 → 可提取
- confidence < 0.85 → skip

对话：
{dialogue}

输出 JSON（只输出 JSON）：
{{"skip": false, "q": "问题", "a": "回答", "tags": ["物流"], "variants": ["变体"], "related": ["退换货政策"], "confidence": 0.92}}
"""

SCORE_PROMPT = """评估这段客服对话的服务质量（0.0-1.0）。

判断信号（从对话本身推断，无需外部标注）：
- 客户说谢谢/好的/明白了/解决了 → 高分
- 对话自然结束，不再追问 → 高分
- 客服给出具体可执行方案 → 高分
- 客户反复追同一问题 → 低分
- 客户语气越来越差 → 低分
- 客服只说"会处理"无后续 → 低分

对话：
{dialogue}

输出 JSON：{{"score": 0.85}}
"""

SOUL_PROMPT = """你是研究人类专家行为的分析师。

以下是筛选出的高质量客服对话。

提炼这些对话背后的"灵魂"——
不是知识点，是让这些对话有效的深层行为模式。

写成 SOUL.md，用第一人称，像一个有经验的人描述自己的工作方式。
必须从对话中归纳，不要凭空编造。

输出格式（严格遵守）：
# SOUL.md

## 我是谁

## 我的沟通本能

## 我怎么处理情绪

## 我的决策直觉

## 我绝不做的事

---
对话样本：
{dialogues}
"""

# ── 工具函数 ──────────────────────────────────────────────────────
def fmt(msgs: list) -> str:
    return "\n".join(
        f"[{m.get('role','?')}] {m.get('message','')}"
        for m in msgs[:20]
    )

def call(prompt: str, json_mode=True) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1 if json_mode else 0.4,
        **kwargs,
    )
    return r.choices[0].message.content.strip()

# ── Pass 1：加载会话 ──────────────────────────────────────────────
def load_sessions(csv_path: str) -> dict[str, list]:
    sessions = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sessions.setdefault(row["session_id"], []).append(row)
    return sessions

# ── Pass 2：并发提取 FAQ ──────────────────────────────────────────
def extract_faq(sid: str, msgs: list, schema: str) -> dict | None:
    try:
        raw = call(FAQ_PROMPT.format(schema=schema, dialogue=fmt(msgs)))
        r = json.loads(raw)
        if r.get("skip") or r.get("confidence", 0) < 0.85:
            return None
        return {**r, "source": sid}
    except Exception:
        return None

def write_faq(entry: dict, base: Path) -> bool:
    slug = hashlib.md5(entry["q"].encode()).hexdigest()[:8]
    path = base / "wiki" / "faq" / f"{slug}.md"
    if path.exists():
        return False
    variants = "\n".join(f"- {v}" for v in entry.get("variants", []))
    related  = "\n".join(f"- [[{r}]]" for r in entry.get("related", []))
    path.write_text(
        f"---\ntype: faq\ntags: {entry.get('tags',[])}\n"
        f"confidence: {entry['confidence']}\nsource: {entry['source']}\n"
        f"updated: {date.today()}\n---\n"
        f"# Q: {entry['q']}\n\n{entry['a']}\n\n"
        f"## 变体问法\n{variants}\n\n## 参见\n{related}\n",
        encoding="utf-8",
    )
    return True

def pass_compile(sessions: dict, base: Path, schema: str, workers=8) -> int:
    print("\n📚 Pass 2 · 提取知识库 FAQ")
    written = 0
    items   = list(sessions.items())

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(extract_faq, sid, msgs, schema): sid
                   for sid, msgs in items}
        done = 0
        for f in as_completed(futures):
            done += 1
            entry = f.result()
            if entry and write_faq(entry, base):
                written += 1
                print(f"  + {entry['q'][:55]}")
            print(f"  [{done}/{len(items)}]", end="\r")

    print(f"  → {written} 条 FAQ 写入               ")
    return written

# ── Pass 3：评分 → 蒸馏 SOUL.md ──────────────────────────────────
def score_session(msgs: list) -> float:
    try:
        r = json.loads(call(SCORE_PROMPT.format(dialogue=fmt(msgs))))
        return float(r.get("score", 0))
    except Exception:
        return 0.0

def pass_distill(sessions: dict, base: Path, workers=8):
    print("\n🧬 Pass 3 · 蒸馏 SOUL.md")
    items = list(sessions.items())

    print(f"  评分 {len(items)} 条会话...")
    scores: list[tuple[float, list]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(score_session, msgs): msgs for _, msgs in items}
        done = 0
        for f in as_completed(futures):
            done += 1
            scores.append((f.result(), futures[f]))
            print(f"  [{done}/{len(items)}]", end="\r")

    scores.sort(key=lambda x: x[0], reverse=True)
    cutoff = max(5, min(30, int(len(scores) * 0.2)))
    top    = [(s, m) for s, m in scores[:cutoff] if s >= 0.70]
    print(f"  → 选出 {len(top)} 条高质量对话（score ≥ 0.70）")

    if len(top) < 3:
        print("  ⚠️  样本不足，跳过 SOUL.md（建议至少 15 条会话）")
        return

    sample = "\n\n".join(
        f"--- 对话 {i+1} (score={s:.2f}) ---\n{fmt(m)}"
        for i, (s, m) in enumerate(top)
    )
    soul = call(SOUL_PROMPT.format(dialogues=sample), json_mode=False)
    (base / "prompts" / "SOUL.md").write_text(soul, encoding="utf-8")
    print(f"  → prompts/SOUL.md 写入完成")

# ── Pass 4：index.md + log.md ─────────────────────────────────────
def pass_meta(csv_path: str, faq_written: int, total: int, base: Path):
    print("\n📋 Pass 4 · 更新索引与日志")

    lines = [f"# Wiki Index\n\n> 更新于 {date.today()}\n\n## FAQ\n\n"]
    for md in sorted((base / "wiki" / "faq").glob("*.md")):
        content = md.read_text(encoding="utf-8")
        q    = next((l.replace("# Q:", "").strip()
                     for l in content.splitlines() if l.startswith("# Q:")), md.stem)
        conf = next((l.split(":")[-1].strip()
                     for l in content.splitlines() if l.startswith("confidence:")), "?")
        lines.append(f"- [[faq/{md.stem}]] {q} `{conf}`\n")
    (base / "wiki" / "index.md").write_text("".join(lines), encoding="utf-8")

    entry = (
        f"\n## [{date.today()}] ingest | {Path(csv_path).name}\n"
        f"- 输入会话：{total} 条\n"
        f"- FAQ 写入：{faq_written} 条\n"
        f"- SOUL.md：{'已生成' if (base / 'prompts' / 'SOUL.md').exists() else '未生成'}\n"
    )
    with open(base / "wiki" / "log.md", "a", encoding="utf-8") as f:
        f.write(entry)

    total_faq = len(list((base / "wiki" / "faq").glob("*.md")))
    print(f"  → wiki/index.md  ({total_faq} 条目)")
    print(f"  → wiki/log.md")

# ── 主入口 ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="从聊天记录生成客服 Agent")
    parser.add_argument("csv",           help="聊天记录 CSV 文件路径")
    parser.add_argument("--workers", "-w", type=int, default=8, help="并发数（默认8）")
    args = parser.parse_args()

    base = Path(".")
    (base / "wiki" / "faq").mkdir(parents=True, exist_ok=True)
    (base / "prompts").mkdir(parents=True, exist_ok=True)

    # schema：优先用 wiki/schema.md，没有则从 templates/ 复制
    schema_dest = base / "wiki" / "schema.md"
    if not schema_dest.exists():
        src = Path("templates/schema.md")
        if src.exists():
            schema_dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    schema = load_schema(base)

    print(f"{'━'*52}")
    print(f"  cs-agent  |  {Path(args.csv).name}")
    print(f"{'━'*52}")

    print("\n📖 Pass 1 · 加载会话")
    sessions = load_sessions(args.csv)
    print(f"  → {len(sessions)} 条会话")

    faq_written = pass_compile(sessions, base, schema, args.workers)
    pass_distill(sessions, base, args.workers)
    pass_meta(args.csv, faq_written, len(sessions), base)

    soul_ok   = (base / "prompts" / "SOUL.md").exists()
    total_faq = len(list((base / "wiki" / "faq").glob("*.md")))

    print(f"\n{'━'*52}")
    print(f"  ✓ 知识库   wiki/faq/        {total_faq} 个页面")
    print(f"  {'✓' if soul_ok else '✗'} 人格      prompts/SOUL.md")
    print(f"  ✓ 索引     wiki/index.md")
    print(f"  ✓ 日志     wiki/log.md")
    print(f"{'━'*52}")
    print(f"  下一步: python sync.py")
    print(f"{'━'*52}\n")

if __name__ == "__main__":
    main()
