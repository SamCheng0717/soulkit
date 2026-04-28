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
