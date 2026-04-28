import sys, os, re, json, random, datetime, argparse, shutil
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from pathlib import Path
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── 客户端 ─────────────────────────────────────────────────────────────────
def _make_advisor_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("ADVISOR_API_KEY") or "placeholder",
        base_url=os.getenv("ADVISOR_BASE_URL", "http://localhost:11434/v1"),
    )

llm_advisor = _make_advisor_client()
ADVISOR_MODEL    = os.getenv("ADVISOR_MODEL", "Qwen/Qwen3.5-27B-FP8")
DINGTALK_WEBHOOK = os.getenv("DINGTALK_WEBHOOK", "")
DINGTALK_SECRET  = os.getenv("DINGTALK_SECRET", "")

PROMPTS_DIR        = Path("prompts")
VERSIONS_DIR       = PROMPTS_DIR / "versions"
CHANGELOG          = PROMPTS_DIR / "CHANGELOG.md"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.md"
CASES_PATH         = Path("tests/cases.json")
FEEDBACK_PATH      = Path("feedback/pending.md")
REPORTS_DIR        = Path("reports")

# ── 解析日报劣质对话区块 ────────────────────────────────────────────────────
_SECTION_RE = re.compile(
    r"### \[会话 ([a-f0-9]+)\].*?得分 [\d.]+.*?用户 `(\w+)`\n"
    r"\*\*问题\*\*：(.+?)\n"
    r"\*\*AI回复\*\*：(.+?)\n"
    r"\*\*建议\*\*：(.+?)(?=\n###|\Z)",
    re.DOTALL,
)

def _parse_bad_sections(report_text: str) -> list[dict]:
    results = []
    for m in _SECTION_RE.finditer(report_text):
        results.append({
            "conv_id":    m.group(1).strip(),
            "user_id":    m.group(2).strip(),
            "problems":   m.group(3).strip(),
            "ai_reply":   m.group(4).strip(),
            "suggestion": m.group(5).strip(),
        })
    return results


def _section_to_case(section: dict, source_date: str = "") -> dict:
    date = source_date
    problems = section.get("problems", "")
    forbidden: list[str] = []
    for kw in ["我们", "案例", "知识库", "保证", "绝对", "百分之百"]:
        if kw in problems or kw in section.get("ai_reply", ""):
            forbidden.append(kw)

    case_id = f"tc_{section['conv_id']}"
    split = "holdout" if random.random() < 0.2 else "optimize"
    return {
        "id":               case_id,
        "split":            split,
        "source":           f"{date}_{section['conv_id']}",
        "customer_input":   section.get("ai_reply", "")[:120],
        "must_not_contain": forbidden,
        "expected_behavior": section.get("suggestion", ""),
    }


def extract_cases(report_path: Path) -> list[dict]:
    text = report_path.read_text(encoding="utf-8")
    date_str = report_path.stem
    sections = _parse_bad_sections(text)

    if CASES_PATH.exists():
        try:
            existing = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []
    else:
        existing = []
    existing_sources = {c["source"] for c in existing}

    new_cases = []
    for sec in sections:
        case = _section_to_case(sec, source_date=date_str)
        if case["source"] not in existing_sources:
            new_cases.append(case)

    if new_cases:
        all_cases = existing + new_cases
        CASES_PATH.parent.mkdir(exist_ok=True)
        CASES_PATH.write_text(json.dumps(all_cases, ensure_ascii=False, indent=2), encoding="utf-8")

    return new_cases


_BEHAVIOR_JUDGE_PROMPT = """\
以下是 AI 客服的一条回复，判断它是否满足要求："{behavior}"
只回答 JSON：{{"ok": true}} 或 {{"ok": false}}

AI回复：
{reply}"""


def _judge_behavior(reply: str, behavior: str) -> bool:
    r = llm_advisor.chat.completions.create(
        model=ADVISOR_MODEL,
        messages=[{"role": "user", "content": _BEHAVIOR_JUDGE_PROMPT.format(
            behavior=behavior, reply=reply
        )}],
        temperature=0.0,
        max_tokens=20,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = (r.choices[0].message.content or "").strip()
    try:
        return bool(json.loads(text).get("ok", False))
    except Exception:
        return False  # 保守策略：解析失败时拒绝，避免误放行


def evaluate_candidate(candidate_prompt: str, cases: list[dict]) -> dict:
    """用候选提示词模拟回复每条用例，返回 {passed, total, passed_count, failures}。"""
    failures = []
    for case in cases:
        # 1. 模拟 AI 回复
        resp = llm_advisor.chat.completions.create(
            model=ADVISOR_MODEL,
            messages=[
                {"role": "system", "content": candidate_prompt},
                {"role": "user",   "content": case["customer_input"]},
            ],
            temperature=0.1,
            max_tokens=256,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        reply = (resp.choices[0].message.content or "").strip()

        # 2. 检查违禁词（精确字符串匹配）
        hit = [w for w in case.get("must_not_contain", []) if w in reply]
        if hit:
            failures.append({
                "id": case["id"], "reason": f"含违禁词：{hit}", "reply": reply[:200]
            })
            continue

        # 3. 检查期望行为（LLM 判断）
        if case.get("expected_behavior"):
            ok = _judge_behavior(reply, case["expected_behavior"])
            if not ok:
                failures.append({
                    "id": case["id"],
                    "reason": f"未满足：{case['expected_behavior']}",
                    "reply": reply[:200],
                })

    total = len(cases)
    passed_count = total - len(failures)
    return {
        "passed":       len(failures) == 0,
        "total":        total,
        "passed_count": passed_count,
        "failures":     failures,
    }


# ── 版本管理 ──────────────────────────────────────────────────────────────────
def get_next_version() -> str:
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(VERSIONS_DIR.glob("v*.md"))
    if not existing:
        return "v001"
    last = existing[-1].stem.split("_")[0]  # "v003"
    n = int(last[1:]) + 1
    return f"v{n:03d}"


def publish_version(candidate: str, version: str, change_info: dict) -> None:
    """归档当前提示词为 version_date.md，写入候选版本，追加 CHANGELOG。"""
    today = datetime.date.today().isoformat()
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 第一次发布时，将当前 system_prompt.md 存为 v000 备份
    if version == "v001" and SYSTEM_PROMPT_PATH.exists():
        v000_files = list(VERSIONS_DIR.glob("v000_*.md"))
        if not v000_files:
            shutil.copy(SYSTEM_PROMPT_PATH, VERSIONS_DIR / f"v000_{today}.md")

    # 将候选版本存为 version_date.md
    archive = VERSIONS_DIR / f"{version}_{today}.md"
    archive.write_text(candidate, encoding="utf-8")

    # 更新 system_prompt.md
    SYSTEM_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_PROMPT_PATH.write_text(candidate, encoding="utf-8")

    # 追加 CHANGELOG
    entry = (
        f"\n## {version} — {today}\n"
        f"**改动模块**：{change_info.get('module', '')}\n"
        f"**原因**：{change_info.get('reason', '')}\n"
        f"**测试**：优化集 {change_info.get('opt_result', '')}，"
        f"验证集 {change_info.get('hold_result', '')}\n"
    )
    CHANGELOG.parent.mkdir(parents=True, exist_ok=True)
    with CHANGELOG.open("a", encoding="utf-8") as f:
        f.write(entry)


def rollback_version(
    version: str,
    versions_dir: Path = VERSIONS_DIR,
    target: Path = SYSTEM_PROMPT_PATH,
) -> None:
    """将 target 恢复为指定版本的内容（rollback vXXX = 使用 vXXX 时发布的候选提示词）。"""
    candidates = list(versions_dir.glob(f"{version}_*.md"))
    if not candidates:
        raise FileNotFoundError(f"版本 {version} 不存在于 {versions_dir}")
    src = sorted(candidates)[-1]
    shutil.copy(src, target)
    print(f"已回滚到 {src.name}")
