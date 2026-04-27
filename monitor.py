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
