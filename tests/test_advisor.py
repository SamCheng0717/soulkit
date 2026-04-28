import sys, json, tempfile
sys.path.insert(0, ".")
from pathlib import Path

SAMPLE_REPORT = """
## 劣质对话详情

### [会话 ad634e](http://example.com) 得分 0.25　用户 `86030`
**问题**：顾客重复追问同一问题、AI 回复出现违禁词（我们）
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
        "ai_reply": "宝宝眼光真好！方便留个微信吗？",
        "suggestion": "直接提供价格范围",
    }
    case = _section_to_case(section, source_date="2026-04-27")
    assert case["id"].startswith("tc_")
    assert case["split"] in ("optimize", "holdout")
    assert case["source"] == "2026-04-27_ad634e"
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
