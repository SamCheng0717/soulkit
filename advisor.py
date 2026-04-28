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
    r"(?:\*\*顾客\*\*：(.+?)\n)?"   # 可选：旧格式日报无此行
    r"\*\*AI回复\*\*：(.+?)\n"
    r"\*\*建议\*\*：(.+?)(?=\n###|\Z)",
    re.DOTALL,
)

def _parse_bad_sections(report_text: str) -> list[dict]:
    results = []
    for m in _SECTION_RE.finditer(report_text):
        results.append({
            "conv_id":       m.group(1).strip(),
            "user_id":       m.group(2).strip(),
            "problems":      m.group(3).strip(),
            "customer_turn": (m.group(4) or "").strip(),
            "ai_reply":      m.group(5).strip(),
            "suggestion":    m.group(6).strip(),
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
        "customer_input":   section.get("customer_turn", "")[:120],
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
    try:
        n = int(last[1:]) + 1
    except ValueError:
        raise ValueError(f"无法解析版本号：{existing[-1].name}，请检查 {VERSIONS_DIR} 目录")
    return f"v{n:03d}"


def publish_version(candidate: str, version: str, change_info: dict) -> None:
    """归档当前提示词为 version_date.md，写入候选版本，追加 CHANGELOG。

    归档语义：vNNN_date.md 存的是该版本的候选内容（非发布前的旧版本）。
    rollback vNNN = 恢复到 vNNN 发布时的提示词内容。
    首次发布时额外保存 v000 备份以保留原始状态。
    """
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


# ── 主管智能体 ────────────────────────────────────────────────────────────────
_SUPERVISOR_PROMPT = """\
你是资深客服主管，负责优化 AI 客服系统提示词。

【当前系统提示词】
{current_prompt}

【今日日报摘要】
{report_text}

【人工反馈】
{feedback_text}

【优化集测试用例（共 {n_cases} 条）】
{cases_text}

{failure_section}

【任务】
1. 分析上述问题，提炼 1 条通用规则（不允许针对具体句子打补丁）
2. 只修改提示词中的一个模块（如"身份规则"/"回复风格"/"违禁词"等区域）
3. 禁止全文重写
4. 输出 JSON（只输出 JSON，不要其他文字）：
{{
  "module": "修改的模块名称",
  "reason": "为什么要改，引用了哪些数据",
  "expected_effect": "改完后预期改善什么",
  "candidate_prompt": "完整的新系统提示词文本"
}}"""


def generate_candidate(
    report_text: str,
    feedback_text: str,
    current_prompt: str,
    optimize_cases: list[dict],
    failures: list[dict] | None = None,
) -> dict:
    cases_text = "\n".join(
        f"- [{c['id']}] 输入：{c['customer_input'][:80]}  期望：{c['expected_behavior']}"
        for c in optimize_cases[:20]
    )
    failure_section = ""
    if failures:
        lines = ["【上次测试失败原因，本次必须修复】"]
        for f in failures[:10]:
            lines.append(f"- {f['id']}: {f['reason']}")
        failure_section = "\n".join(lines)

    prompt = _SUPERVISOR_PROMPT.format(
        current_prompt=current_prompt,
        report_text=report_text[:3000],
        feedback_text=feedback_text[:1000] if feedback_text else "（无）",
        n_cases=len(optimize_cases),
        cases_text=cases_text,
        failure_section=failure_section,
    )
    r = llm_advisor.chat.completions.create(
        model=ADVISOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4096,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = (r.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        return {
            "module": "未知",
            "reason": "解析失败",
            "expected_effect": "",
            "candidate_prompt": current_prompt,
        }


# ── 主循环 ────────────────────────────────────────────────────────────────────
MAX_RETRIES = 3
HOLDOUT_PASS_THRESHOLD = 0.75


def _load_cases() -> tuple[list[dict], list[dict]]:
    if CASES_PATH.exists():
        try:
            all_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            all_cases = []
    else:
        all_cases = []
    optimize = [c for c in all_cases if c.get("split") == "optimize"]
    holdout  = [c for c in all_cases if c.get("split") == "holdout"]
    return optimize, holdout


def run_advisor(
    report_path: Path,
    extract_only: bool = False,
    rollback: str = "",
) -> dict:
    if rollback:
        rollback_version(rollback)
        return {"action": "rolled_back", "version": rollback}

    new_cases = extract_cases(report_path)
    print(f"  → 新增测试用例 {len(new_cases)} 条")

    if extract_only:
        return {"action": "extracted", "new_cases": len(new_cases)}

    report_text    = report_path.read_text(encoding="utf-8")
    current_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    feedback_text  = FEEDBACK_PATH.read_text(encoding="utf-8") if FEEDBACK_PATH.exists() else ""
    optimize_cases, holdout_cases = _load_cases()

    if not optimize_cases:
        print("  → 优化集为空，跳过本轮优化")
        return {"action": "skipped", "reason": "no_cases"}

    failures = None
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n  [优化 {attempt}/{MAX_RETRIES}] 生成候选提示词...")
        result = generate_candidate(
            report_text, feedback_text, current_prompt, optimize_cases, failures
        )
        candidate = result.get("candidate_prompt", current_prompt)

        print(f"  → 评估优化集（{len(optimize_cases)} 条）...")
        opt_eval = evaluate_candidate(candidate, optimize_cases)
        print(f"     优化集：{opt_eval['passed_count']}/{opt_eval['total']}")

        if not opt_eval["passed"]:
            failures = opt_eval["failures"]
            print(f"     ✗ 优化集未通过，失败 {len(failures)} 条")
            continue

        print(f"  → 评估验证集（{len(holdout_cases)} 条）...")
        if holdout_cases:
            hold_eval = evaluate_candidate(candidate, holdout_cases)
            hold_rate = hold_eval["passed_count"] / hold_eval["total"]
            print(f"     验证集：{hold_eval['passed_count']}/{hold_eval['total']}（{hold_rate:.0%}）")
            if hold_rate < HOLDOUT_PASS_THRESHOLD:
                failures = hold_eval["failures"]
                print(f"     ✗ 验证集通过率 {hold_rate:.0%} < {HOLDOUT_PASS_THRESHOLD:.0%}，重试")
                continue
            hold_result = f"{hold_eval['passed_count']}/{hold_eval['total']}"
        else:
            hold_result = "N/A（验证集为空）"

        version = get_next_version()
        publish_version(
            candidate=candidate,
            version=version,
            change_info={
                "module":      result.get("module", ""),
                "reason":      result.get("reason", ""),
                "opt_result":  f"{opt_eval['passed_count']}/{opt_eval['total']}",
                "hold_result": hold_result,
            }
        )
        print(f"  → 发布 {version} 成功！模块：{result.get('module')}")

        # 仅在成功发布时清空反馈，失败时保留供下次重试或人工介入
        if FEEDBACK_PATH.exists() and feedback_text.strip():
            FEEDBACK_PATH.write_text("", encoding="utf-8")

        return {
            "action":      "published",
            "version":     version,
            "module":      result.get("module"),
            "reason":      result.get("reason"),
            "opt_result":  f"{opt_eval['passed_count']}/{opt_eval['total']}",
            "hold_result": hold_result,
        }

    print(f"  → {MAX_RETRIES} 次重试全失败，本轮放弃")
    return {
        "action":   "failed",
        "attempts": MAX_RETRIES,
        "failures": failures or [],
    }


# ── 钉钉通知 ──────────────────────────────────────────────────────────────────
def send_advisor_dingtalk(result: dict, date: str) -> None:
    import time, hmac, hashlib, base64, urllib.parse
    if not DINGTALK_WEBHOOK or not DINGTALK_SECRET:
        return

    action = result.get("action")
    if action == "published":
        title = f"提示词更新 {result['version']} — {date}"
        lines = [
            f"## {title}", "",
            f"**改动模块**：{result.get('module', '')}",
            f"**原因**：{result.get('reason', '')[:200]}",
            f"**测试**：优化集 {result.get('opt_result')}，验证集 {result.get('hold_result')}",
        ]
    elif action == "failed":
        title = f"提示词优化未通过 — {date}"
        top = result.get("failures", [])[:3]
        lines = [
            f"## {title}", "",
            f"**{result.get('attempts')} 次重试全失败**",
            "**失败原因（前3条）：**",
        ] + [f"- {f['id']}: {f['reason']}" for f in top] + [
            "", "建议：请检查 feedback/pending.md 并手动调整",
        ]
    elif action == "rolled_back":
        title = f"提示词已回滚 {result.get('version')} — {date}"
        lines = [f"## {title}", "", f"已回滚到版本 {result.get('version')}"]
    else:
        return

    text = "\n".join(lines)
    ts       = str(round(time.time() * 1000))
    sign_src = f"{ts}\n{DINGTALK_SECRET}".encode("utf-8")
    digest   = hmac.new(DINGTALK_SECRET.encode("utf-8"), sign_src, digestmod=hashlib.sha256).digest()
    sign     = urllib.parse.quote_plus(base64.b64encode(digest))
    url      = f"{DINGTALK_WEBHOOK}&timestamp={ts}&sign={sign}"
    resp = requests.post(url, json={
        "msgtype":  "markdown",
        "markdown": {"title": title, "text": text},
    }, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errcode") != 0:
        raise RuntimeError(data.get("errmsg", str(data)))


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BeautsGO 客服主管智能体")
    parser.add_argument("--report",       default="",  help="指定日报路径，默认取最新")
    parser.add_argument("--extract-only", action="store_true", help="只提取测试用例，不优化")
    parser.add_argument("--rollback",     default="",  help="回滚到指定版本，如 v002")
    args = parser.parse_args()

    date_str = datetime.date.today().isoformat()
    print(f"{'='*52}")
    print(f"  BeautsGO Advisor  |  {date_str}")
    print(f"{'='*52}\n")

    if args.rollback:
        result = run_advisor(report_path=Path("."), rollback=args.rollback)
    else:
        if args.report:
            report_path = Path(args.report)
        else:
            reports = sorted(REPORTS_DIR.glob("????-??-??.md"))
            if not reports:
                print("错误：reports/ 下无日报文件")
                return
            report_path = reports[-1]
        print(f"  日报：{report_path}")
        result = run_advisor(report_path, extract_only=args.extract_only)

    print(f"\n  结果：{result['action']}")

    try:
        send_advisor_dingtalk(result, date_str)
        print("  → 钉钉推送成功")
    except Exception as e:
        print(f"  ⚠ 钉钉推送失败：{e}")

    print(f"\n{'='*52}\n")


if __name__ == "__main__":
    main()
