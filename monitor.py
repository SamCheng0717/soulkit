# monitor.py
import os, json, datetime, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── 客户端 ────────────────────────────────────────────────────────────────
def _make_clients():
    ds = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY") or "placeholder",
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    local_key = os.getenv("LOCAL_API_KEY") or "placeholder"
    local_base = os.getenv("LOCAL_BASE_URL") or "http://localhost:11434/v1"
    local = OpenAI(api_key=local_key, base_url=local_base)
    return ds, local

llm_ds, llm_local = _make_clients()

DIFY_BASE   = os.getenv("DIFY_BASE_URL", "http://localhost:80").rstrip("/")
DIFY_KEY    = os.getenv("DIFY_API_KEY", "")
DIFY_USER   = os.getenv("DIFY_USER", "monitor")
DS_MODEL    = os.getenv("MODEL", "deepseek-chat")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen3-14B-AWQ")

REPORTS = Path("reports")
STATS   = REPORTS / "stats.json"


# ── Dify API ──────────────────────────────────────────────────────────────
def fetch_conversations(since: datetime.datetime) -> list[dict]:
    """拉取 since 之后创建的所有对话（分页）。"""
    # since 必须是 naive datetime（datetime.datetime.now()），与 fromtimestamp 保持一致
    headers = {"Authorization": f"Bearer {DIFY_KEY}"}
    results, last_id = [], None
    while True:
        params: dict = {"user": DIFY_USER, "limit": 100, "sort_by": "-updated_at"}
        if last_id:
            params["last_id"] = last_id
        r = requests.get(f"{DIFY_BASE}/v1/conversations", headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        convs = data.get("data", [])
        if not convs:
            break
        for c in convs:
            created = datetime.datetime.fromtimestamp(c["created_at"])
            if created < since:
                return results
            results.append(c)
        if not data.get("has_more"):
            break
        last_id = convs[-1]["id"]
    return results


def fetch_messages(conv_id: str) -> list[dict]:
    """拉取单条对话的完整消息记录（分页）。"""
    headers = {"Authorization": f"Bearer {DIFY_KEY}"}
    results, first_id = [], None
    while True:
        params: dict = {"user": DIFY_USER, "conversation_id": conv_id, "limit": 100}
        if first_id:
            params["first_id"] = first_id
        r = requests.get(f"{DIFY_BASE}/v1/messages", headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        msgs = data.get("data", [])
        results.extend(msgs)
        if not data.get("has_more"):
            break
        if msgs:
            first_id = msgs[0]["id"]
        else:
            break
    results.reverse()
    return results


def format_dialogue(messages: list[dict]) -> str:
    """将消息列表格式化为 LLM 可读的对话文本。"""
    lines = []
    for m in messages:
        if m.get("query"):
            lines.append(f"[顾客] {m['query']}")
        if m.get("answer"):
            lines.append(f"[AI] {m['answer']}")
    return "\n".join(lines)


# ── 留资检测 (Qwen3-14B 本地) ──────────────────────────────────────────────
CONVERSION_PROMPT = """\
以下是一段客服对话，判断顾客是否留下了微信号或手机号。
只回答 JSON：{{"留资": true}} 或 {{"留资": false}}

对话：
{dialogue}"""


def detect_conversion(dialogue: str) -> bool:
    """用本地 Qwen3-14B 判断对话是否有顾客留资。"""
    r = llm_local.chat.completions.create(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": CONVERSION_PROMPT.format(dialogue=dialogue)}],
        temperature=0.1,
        max_tokens=20,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = (r.choices[0].message.content or "").strip()
    try:
        return bool(json.loads(text).get("留资", False))
    except Exception:
        return "true" in text.lower()


# ── 质量评分 (DeepSeek) ────────────────────────────────────────────────────
SCORE_PROMPT = """\
分析以下客服对话，判断 AI 回复质量。

检测信号：
- 顾客出现负面情绪词（没用/烦/算了/什么破）→ 扣分
- 顾客重复问同一问题 2+ 次 → 扣分
- AI 回复后顾客沉默离开 → 扣分
- AI 回复出现违禁词（我们/案例/知识库/保证）→ 扣分
- 顾客全程未留微信/电话 → 轻微扣分

输出 JSON（只输出 JSON）：
{{"score": 0.35, "problems": ["重复追问"], "bad_turn": "AI第3条回复原文", "suggestion": "建议内容"}}

对话：
{dialogue}"""


def score_conversation(dialogue: str) -> dict:
    """用 DeepSeek 对对话质量打分并生成诊断建议。"""
    r = llm_ds.chat.completions.create(
        model=DS_MODEL,
        messages=[{"role": "user", "content": SCORE_PROMPT.format(dialogue=dialogue)}],
        temperature=0.1,
        max_tokens=512,
    )
    text = (r.choices[0].message.content or "").strip()
    try:
        return json.loads(text)
    except Exception:
        return {"score": 1.0, "problems": [], "bad_turn": "", "suggestion": ""}


# ── 统计持久化 ──────────────────────────────────────────────────────────────
def load_stats() -> list[dict]:
    if not STATS.exists():
        return []
    return json.loads(STATS.read_text(encoding="utf-8"))


def append_stats(date: str, total: int, converted: int, bad: int) -> None:
    stats = load_stats()
    stats = [s for s in stats if s["date"] != date]
    rate  = round(converted / total, 3) if total else 0.0
    stats.append({"date": date, "total": total, "converted": converted, "rate": rate, "bad": bad})
    stats.sort(key=lambda x: x["date"])
    REPORTS.mkdir(exist_ok=True)
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 日报 ────────────────────────────────────────────────────────────────────
def generate_daily_report(date: str, results: list[dict], threshold: float = 0.6) -> Path:
    total     = len(results)
    converted = sum(1 for r in results if r["converted"])
    bad       = [r for r in results if r["score"]["score"] < threshold]
    rate      = f"{converted / total * 100:.1f}%" if total else "0%"

    problems: dict[str, int] = {}
    for r in bad:
        for p in r["score"].get("problems", []):
            problems[p] = problems.get(p, 0) + 1

    lines = [
        f"# 日报 {date}\n",
        f"对话总量：{total} 条 | 留资：{converted} 条 | 留资率：{rate}",
        f"劣质对话：{len(bad)} 条 | 阈值：{threshold}\n",
    ]

    if problems:
        lines.append("## 问题分布\n")
        for p, c in sorted(problems.items(), key=lambda x: -x[1]):
            lines.append(f"- {p}：{c} 条")
        lines.append("")

    if bad:
        lines.append("\n## 劣质对话详情\n")
        for r in bad:
            s = r["score"]
            probs = "、".join(s.get("problems", [])) or "未分类"
            lines += [
                f"### [会话 {r['id'][:6]}] 得分 {s['score']:.2f}",
                f"**问题**：{probs}",
                f"**AI回复**：{s.get('bad_turn', '')}",
                f"**建议**：{s.get('suggestion', '')}\n",
            ]

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{date}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ── 周报 ────────────────────────────────────────────────────────────────────
def generate_weekly_report() -> Path:
    stats = load_stats()
    today = datetime.date.today()
    monday    = today - datetime.timedelta(days=today.weekday())
    sunday    = monday + datetime.timedelta(days=6)
    prev_mon  = monday - datetime.timedelta(days=7)
    prev_sun  = monday - datetime.timedelta(days=1)

    week_data = [s for s in stats if monday.isoformat() <= s["date"] <= sunday.isoformat()]
    prev_data = [s for s in stats if prev_mon.isoformat() <= s["date"] <= prev_sun.isoformat()]

    if not week_data:
        raise ValueError("本周暂无数据，请先运行日报")

    total     = sum(s["total"]     for s in week_data)
    converted = sum(s["converted"] for s in week_data)
    rate      = converted / total if total else 0.0

    prev_total     = sum(s["total"]     for s in prev_data)
    prev_converted = sum(s["converted"] for s in prev_data)
    prev_rate      = prev_converted / prev_total if prev_total else 0.0

    diff   = rate - prev_rate
    arrow  = "↑" if diff > 0 else "↓" if diff < 0 else "→"
    diff_s = f"+{diff*100:.1f}%" if diff > 0 else f"{diff*100:.1f}%"

    lines = [
        f"# 周报 {monday} ~ {sunday}\n",
        "## 留资率趋势\n",
        "| 日期 | 对话量 | 留资量 | 留资率 |",
        "|------|--------|--------|--------|",
    ]
    for s in week_data:
        lines.append(f"| {s['date']} | {s['total']} | {s['converted']} | {s['rate']*100:.1f}% |")
    lines.append(f"| **合计** | **{total}** | **{converted}** | **{rate*100:.1f}%** |\n")
    if prev_data:
        lines.append(f"趋势：本周留资率较上周 {diff_s} {arrow}\n")
    else:
        lines.append("趋势：上周暂无对比数据\n")

    week_num = today.isocalendar()[1]
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"week-{today.year}-{week_num:02d}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ── 单条对话处理（用于并发）──────────────────────────────────────────────────
def process_conversation(conv_id: str, dialogue: str) -> dict:
    return {
        "id":        conv_id,
        "converted": detect_conversion(dialogue),
        "score":     score_conversation(dialogue),
    }


# ── 主入口 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BeautsGO AI 对话质量监控")
    parser.add_argument("--since",     default="24h",    help="时间范围，如 24h / 48h")
    parser.add_argument("--threshold", type=float, default=0.6, help="劣质判断阈值")
    parser.add_argument("--report",    choices=["daily", "weekly", "both"], default="daily")
    parser.add_argument("--workers",   type=int,   default=8)
    args = parser.parse_args()

    try:
        hours = int(args.since.replace("h", ""))
    except ValueError:
        print(f"错误：--since 格式不正确（'{args.since}'），请使用如 24h / 48h")
        return
    since_dt = datetime.datetime.now() - datetime.timedelta(hours=hours)
    date_str = datetime.date.today().isoformat()

    print(f"{'='*52}")
    print(f"  BeautsGO Monitor  |  {args.since}内  |  阈值 {args.threshold}")
    print(f"{'='*52}")

    print(f"\n[1/4] 拉取对话...")
    convs = fetch_conversations(since_dt)
    print(f"  → {len(convs)} 条对话")
    if not convs:
        print("  无数据，退出。")
        return

    print(f"\n[2/4] 拉取消息记录...")
    dialogues: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_messages, c["id"]): c["id"] for c in convs}
        for f in as_completed(futures):
            cid = futures[f]
            try:
                msgs = f.result()
                dialogues[cid] = format_dialogue(msgs)
            except Exception as e:
                print(f"  ⚠ 拉取对话 {cid[:8]} 消息失败：{e}")
    print(f"  → {len(dialogues)} 条消息记录拉取完成")

    print(f"\n[3/4] 留资检测 + 质量评分...")
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures2 = {
            ex.submit(process_conversation, cid, dia): cid
            for cid, dia in dialogues.items()
        }
        done = 0
        for f in as_completed(futures2):
            done += 1
            try:
                results.append(f.result())
            except Exception as e:
                print(f"\n  ⚠ 处理对话失败：{e}")
            print(f"  [{done}/{len(dialogues)}]", end="\r")
    print()

    total     = len(results)
    converted = sum(1 for r in results if r["converted"])
    bad_count = sum(1 for r in results if r["score"]["score"] < args.threshold)
    print(f"  → 留资率 {converted/total*100:.1f}%  |  劣质 {bad_count} 条")

    print(f"\n[4/4] 生成报告...")
    append_stats(date_str, total, converted, bad_count)

    if args.report in ("daily", "both"):
        path = generate_daily_report(date_str, results, args.threshold)
        print(f"  → 日报: {path}")
    if args.report in ("weekly", "both"):
        try:
            path = generate_weekly_report()
            print(f"  → 周报: {path}")
        except ValueError as e:
            print(f"  ⚠ 周报跳过：{e}")

    print(f"\n{'='*52}\n")


if __name__ == "__main__":
    main()
