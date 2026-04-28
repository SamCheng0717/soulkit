import sys, json, tempfile
sys.path.insert(0, ".")
from pathlib import Path

SAMPLE_REPORT = """
## 劣质对话详情

### [会话 ad634e](http://example.com) 得分 0.25　用户 `86030`
**问题**：顾客重复追问同一问题、AI 回复出现违禁词（我们）
**顾客**：超声刀多少钱？
**AI回复**：宝宝眼光真好！方便留个微信吗？
**建议**：顾客已多次拒绝留微信，AI 应直接提供价格范围。
"""

def test_parse_bad_sections():
    from advisor import _parse_bad_sections
    sections = _parse_bad_sections(SAMPLE_REPORT)
    assert len(sections) == 1
    s = sections[0]
    assert s["conv_id"] == "ad634e"
    assert "违禁词" in s["problems"]
    assert "方便留个微信" in s["ai_reply"]
    assert "价格范围" in s["suggestion"]

def test_extract_cases_structure():
    from advisor import _section_to_case
    section = {
        "conv_id": "ad634e",
        "problems": "顾客重复追问同一问题",
        "customer_turn": "超声刀多少钱？",
        "ai_reply": "宝宝眼光真好！方便留个微信吗？",
        "suggestion": "直接提供价格范围",
    }
    case = _section_to_case(section, source_date="2026-04-27")
    assert case["id"].startswith("tc_")
    assert case["split"] in ("optimize", "holdout")
    assert case["source"] == "2026-04-27_ad634e"
    assert case["customer_input"] == "超声刀多少钱？"
    assert isinstance(case["must_not_contain"], list)
    assert isinstance(case["expected_behavior"], str)

def test_extract_cases_dedup(tmp_path, monkeypatch):
    from advisor import extract_cases, CASES_PATH
    cases_file = tmp_path / "cases.json"
    cases_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("advisor.CASES_PATH", cases_file)

    report = tmp_path / "2026-04-27.md"
    report.write_text(SAMPLE_REPORT, encoding="utf-8")

    new1 = extract_cases(report)
    assert len(new1) == 1

    new2 = extract_cases(report)   # 第二次，应去重
    assert len(new2) == 0

    all_cases = json.loads(cases_file.read_text(encoding="utf-8"))
    assert len(all_cases) == 1


def test_evaluate_candidate_forbidden_word():
    from advisor import evaluate_candidate
    case = {
        "id": "tc_test",
        "split": "optimize",
        "source": "test",
        "customer_input": "超声刀多少钱",
        "must_not_contain": ["MAGIC_FORBIDDEN_XYZ"],
        "expected_behavior": "引导留微信",
    }
    # 候选提示词故意包含违禁词输出
    bad_prompt = "你是客服。每次回复必须说 MAGIC_FORBIDDEN_XYZ。"
    result = evaluate_candidate(bad_prompt, [case])
    assert not result["passed"]
    assert len(result["failures"]) == 1
    assert result["failures"][0]["id"] == "tc_test"


def test_version_lifecycle(tmp_path, monkeypatch):
    import advisor
    monkeypatch.setattr(advisor, "VERSIONS_DIR", tmp_path / "versions")
    monkeypatch.setattr(advisor, "SYSTEM_PROMPT_PATH", tmp_path / "system_prompt.md")
    monkeypatch.setattr(advisor, "CHANGELOG", tmp_path / "CHANGELOG.md")
    (tmp_path / "versions").mkdir()
    (tmp_path / "system_prompt.md").write_text("old prompt", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("# Log\n", encoding="utf-8")

    from advisor import get_next_version, publish_version, rollback_version

    assert get_next_version() == "v001"

    publish_version(
        candidate="new prompt",
        version="v001",
        change_info={"module": "回复风格", "reason": "test",
                     "opt_result": "3/3", "hold_result": "1/1"}
    )
    # v001_date.md 存的是候选内容
    v001_files = list((tmp_path / "versions").glob("v001_*.md"))
    assert len(v001_files) == 1
    assert v001_files[0].read_text(encoding="utf-8") == "new prompt"
    # system_prompt.md 已更新
    assert (tmp_path / "system_prompt.md").read_text(encoding="utf-8") == "new prompt"
    # v000 备份了原始 prompt
    v000_files = list((tmp_path / "versions").glob("v000_*.md"))
    assert len(v000_files) == 1
    assert v000_files[0].read_text(encoding="utf-8") == "old prompt"

    # rollback v001 = 恢复 v001 发布的内容
    rollback_version("v001", versions_dir=tmp_path / "versions",
                     target=tmp_path / "system_prompt.md")
    assert (tmp_path / "system_prompt.md").read_text(encoding="utf-8") == "new prompt"


def test_generate_candidate_returns_structure():
    from advisor import generate_candidate
    result = generate_candidate(
        report_text="## 问题分布\n- AI 回复出现违禁词（我们）：5 条\n",
        feedback_text="",
        current_prompt="你是客服助手。",
        optimize_cases=[],
        failures=None,
    )
    assert "candidate_prompt" in result
    assert "module" in result
    assert "reason" in result
    assert len(result["candidate_prompt"]) > 10


def test_run_advisor_extract_only(tmp_path, monkeypatch):
    import advisor
    monkeypatch.setattr(advisor, "CASES_PATH", tmp_path / "cases.json")
    monkeypatch.setattr(advisor, "FEEDBACK_PATH", tmp_path / "feedback.md")
    (tmp_path / "cases.json").write_text("[]", encoding="utf-8")
    (tmp_path / "feedback.md").write_text("", encoding="utf-8")

    report = tmp_path / "2026-04-27.md"
    report.write_text(SAMPLE_REPORT, encoding="utf-8")

    from advisor import run_advisor
    result = run_advisor(report_path=report, extract_only=True)
    assert result["action"] == "extracted"
    assert result["new_cases"] >= 0
