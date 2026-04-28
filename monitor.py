# monitor.py
import sys, os, json, datetime, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
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

DIFY_BASE          = os.getenv("DIFY_BASE_URL", "http://localhost:80").rstrip("/")
DIFY_KEY           = os.getenv("DIFY_API_KEY", "")
DIFY_APP_ID        = os.getenv("DIFY_APP_ID", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "")
DS_MODEL           = os.getenv("MODEL", "deepseek-chat")
LOCAL_MODEL        = os.getenv("LOCAL_MODEL", "Qwen/Qwen3-14B-AWQ")
DINGTALK_WEBHOOK   = os.getenv("DINGTALK_WEBHOOK", "")
DINGTALK_SECRET    = os.getenv("DINGTALK_SECRET", "")

REPORTS = Path("reports")
STATS   = REPORTS / "stats.json"


# ── DB + Dify App API ────────────────────────────────────────────────────
def _db_conn():
    import pymysql
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASS, database=DB_NAME,
        charset="utf8mb4", connect_timeout=10,
    )


def _get_all_member_ids() -> list[str]:
    """从 DB 拉取服务群（group_category=2）顾客 member_id。
    Dify 实际使用的 user 参数就是 hd_group_member.id。
    只取 id <= 90000 的范围（超出范围的群不使用此 Dify App）。"""
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT gm.id "
                "FROM hd_group_member gm "
                "JOIN hd_groups g ON g.gid = gm.gid "
                "WHERE g.group_category = 2 "
                "AND gm.id <= 90000 "
                "AND gm.delete_time IS NULL "
                "AND g.delete_time IS NULL "
                "ORDER BY gm.id DESC"
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def _fetch_convs_for_uid(session: "requests.Session", uid: str, since_ts: float) -> list[dict]:
    """用 App API 拉取单个用户 since 之后更新的对话列表（复用 session）。"""
    results, last_id = [], None
    while True:
        params: dict = {"user": uid, "limit": 100}
        if last_id:
            params["last_id"] = last_id
        r = session.get(f"{DIFY_BASE}/v1/conversations", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        convs = data.get("data", [])
        if not convs:
            break
        for c in convs:
            if c.get("updated_at", 0) < since_ts:
                return results
            c["_uid"] = uid
            results.append(c)
        if not data.get("has_more"):
            break
        last_id = convs[-1]["id"]
    return results


def fetch_conversations(since: datetime.datetime) -> list[dict]:
    """DB 查所有服务群 member_id → 并发问 Dify → 只保留 since 后有更新的对话。"""
    import threading
    member_ids = _get_all_member_ids()
    print(f"  → 共 {len(member_ids)} 个 member_id，并发查询 Dify...")
    since_ts = since.timestamp()
    seen: dict[str, dict] = {}
    _local = threading.local()

    def _get_session():
        if not hasattr(_local, "session"):
            s = requests.Session()
            s.headers["Authorization"] = f"Bearer {DIFY_KEY}"
            _local.session = s
        return _local.session

    def _fetch(uid):
        return _fetch_convs_for_uid(_get_session(), uid, since_ts)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_fetch, uid): uid for uid in member_ids}
        done = 0
        for f in as_completed(futures):
            done += 1
            if done % 500 == 0:
                print(f"  [{done}/{len(member_ids)}] 已找到 {len(seen)} 条对话", end="\r")
            try:
                for c in f.result():
                    seen[c["id"]] = c
            except Exception:
                pass
    return list(seen.values())


def fetch_messages(conv_id: str, user_id: str) -> list[dict]:
    """拉取单条对话的完整消息记录（App API + from_end_user_session_id）。"""
    headers = {"Authorization": f"Bearer {DIFY_KEY}"}
    results, first_id = [], None
    while True:
        params: dict = {"user": user_id, "conversation_id": conv_id, "limit": 100}
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
{{"score": 0.35, "problems": ["重复追问"], "customer_turn": "触发该问题的顾客原话", "bad_turn": "AI第3条回复原文", "suggestion": "建议内容"}}

对话：
{dialogue}"""


def score_conversation(dialogue: str) -> dict:
    """用 DeepSeek 对对话质量打分并生成诊断建议。"""
    r = llm_ds.chat.completions.create(
        model=DS_MODEL,
        messages=[{"role": "user", "content": SCORE_PROMPT.format(dialogue=dialogue)}],
        temperature=0.1,
        max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = (r.choices[0].message.content or "").strip()
    # 去掉 ```json ... ``` 包裹
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
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


# ── 钉钉推送 ──────────────────────────────────────────────────────────────────
def send_dingtalk(date: str, results: list[dict], threshold: float, logs_url: str) -> None:
    """将日报统计摘要推送到钉钉群机器人。"""
    import time, hmac, hashlib, base64, urllib.parse

    if not DINGTALK_WEBHOOK or not DINGTALK_SECRET:
        return

    total     = len(results)
    converted = sum(1 for r in results if r["converted"])
    bad       = [r for r in results if r["score"]["score"] < threshold]
    rate      = f"{converted / total * 100:.1f}%" if total else "0%"

    problems: dict[str, int] = {}
    for r in bad:
        for p in r["score"].get("problems", []):
            problems[p] = problems.get(p, 0) + 1
    top5 = sorted(problems.items(), key=lambda x: -x[1])[:5]

    lines = [
        f"## BeautsGO 日报 {date}",
        "",
        f"**对话量** {total} 条 ｜ **留资** {converted} 条 ｜ **留资率** {rate}",
        f"**劣质对话** {len(bad)} 条（阈值 {threshold}）",
        "",
    ]
    if top5:
        lines.append("**Top 问题：**")
        for p, c in top5:
            lines.append(f"- {p}：{c} 条")
        lines.append("")
    if logs_url:
        lines.append(f"[查看 Dify 日志]({logs_url})")

    text = "\n".join(lines)

    # HMAC-SHA256 签名
    ts       = str(round(time.time() * 1000))
    sign_src = f"{ts}\n{DINGTALK_SECRET}".encode("utf-8")
    digest   = hmac.new(DINGTALK_SECRET.encode("utf-8"), sign_src, digestmod=hashlib.sha256).digest()
    sign     = urllib.parse.quote_plus(base64.b64encode(digest))
    url      = f"{DINGTALK_WEBHOOK}&timestamp={ts}&sign={sign}"

    resp = requests.post(url, json={
        "msgtype":  "markdown",
        "markdown": {"title": f"BeautsGO 日报 {date}", "text": text},
    }, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errcode") != 0:
        raise RuntimeError(data.get("errmsg", str(data)))


# ── 日报 ────────────────────────────────────────────────────────────────────
def generate_daily_report(
    date: str, results: list[dict], threshold: float = 0.6,
    conv_url_base: str = ""
) -> Path:
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
            conv_id = r["id"]
            user_id = r.get("user_id", "")
            # 链接到日志列表页；右侧括注用户 UID 供 Dify 搜索框定位
            if conv_url_base:
                heading = f"### [会话 {conv_id[:6]}]({conv_url_base}) 得分 {s['score']:.2f}　用户 `{user_id}`"
            else:
                heading = f"### [会话 {conv_id[:6]}] 得分 {s['score']:.2f}"
            lines += [
                heading,
                f"**问题**：{probs}",
                f"**顾客**：{s.get('customer_turn', '')}",
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
def process_conversation(conv_id: str, dialogue: str, user_id: str = "") -> dict:
    return {
        "id":        conv_id,
        "user_id":   user_id,
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
    # {conv_id: (user_id, dialogue)}
    dialogues: dict[str, tuple[str, str]] = {}
    conv_users = {c["id"]: c.get("_uid", "") for c in convs}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_messages, c["id"], conv_users[c["id"]]): c["id"] for c in convs}
        for f in as_completed(futures):
            cid = futures[f]
            try:
                msgs = f.result()
                dialogues[cid] = (conv_users[cid], format_dialogue(msgs))
            except Exception as e:
                print(f"  ⚠ 拉取对话 {cid[:8]} 消息失败：{e}")
    print(f"  → {len(dialogues)} 条消息记录拉取完成")

    print(f"\n[3/4] 留资检测 + 质量评分...")
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures2 = {
            ex.submit(process_conversation, cid, dia, uid): cid
            for cid, (uid, dia) in dialogues.items()
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
    rate_str  = f"{converted/total*100:.1f}%" if total else "N/A"
    print(f"  → 留资率 {rate_str}  |  劣质 {bad_count} 条")

    if not total:
        print("  所有对话处理失败，退出。")
        return

    print(f"\n[4/4] 生成报告...")
    append_stats(date_str, total, converted, bad_count)

    conv_url_base = f"{DIFY_BASE}/app/{DIFY_APP_ID}/logs" if DIFY_APP_ID else ""
    if args.report in ("daily", "both"):
        path = generate_daily_report(date_str, results, args.threshold, conv_url_base)
        print(f"  → 日报: {path}")
    if args.report in ("weekly", "both"):
        try:
            path = generate_weekly_report()
            print(f"  → 周报: {path}")
        except ValueError as e:
            print(f"  ⚠ 周报跳过：{e}")

    if DINGTALK_WEBHOOK:
        try:
            send_dingtalk(date_str, results, args.threshold, conv_url_base)
            print("  → 钉钉推送成功")
        except Exception as e:
            print(f"  ⚠ 钉钉推送失败：{e}")

    print(f"\n{'='*52}\n")


if __name__ == "__main__":
    main()
