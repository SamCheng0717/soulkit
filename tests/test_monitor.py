import json, sys, unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

sys.path.insert(0, "E:/cs-agent")

class TestDifyClient(unittest.TestCase):

    def test_format_dialogue_interleaves_messages(self):
        from monitor import format_dialogue
        messages = [
            {"query": "超声刀多少钱", "answer": "宝宝留个微信～"},
            {"query": "你没回答我", "answer": "亲爱的价格区间在2100左右"},
        ]
        result = format_dialogue(messages)
        self.assertIn("[顾客] 超声刀多少钱", result)
        self.assertIn("[AI] 宝宝留个微信～", result)
        self.assertIn("[顾客] 你没回答我", result)

    def test_format_dialogue_skips_empty(self):
        from monitor import format_dialogue
        messages = [{"query": "你好", "answer": ""}]
        result = format_dialogue(messages)
        self.assertIn("[顾客] 你好", result)
        self.assertNotIn("[AI]", result)

    @patch("monitor.requests.get")
    def test_fetch_conversations_filters_by_since(self, mock_get):
        from monitor import fetch_conversations
        now = datetime.now()
        old_ts = int((now - timedelta(hours=48)).timestamp())
        new_ts = int((now - timedelta(hours=1)).timestamp())
        mock_get.return_value.json.return_value = {
            "data": [
                {"id": "new1", "created_at": new_ts},
                {"id": "old1", "created_at": old_ts},
            ],
            "has_more": False,
        }
        mock_get.return_value.raise_for_status = MagicMock()
        since = now - timedelta(hours=24)
        result = fetch_conversations(since)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "new1")


class TestConversionDetection(unittest.TestCase):

    @patch("monitor.llm_local")
    def test_detect_conversion_true(self, mock_llm):
        from monitor import detect_conversion
        mock_llm.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='{"留资": true}'))
        ]
        self.assertTrue(detect_conversion("[顾客] 我微信是 abc123"))

    @patch("monitor.llm_local")
    def test_detect_conversion_false(self, mock_llm):
        from monitor import detect_conversion
        mock_llm.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='{"留资": false}'))
        ]
        self.assertFalse(detect_conversion("[顾客] 超声刀多少钱"))

    @patch("monitor.llm_local")
    def test_detect_conversion_fallback_on_bad_json(self, mock_llm):
        from monitor import detect_conversion
        mock_llm.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="true"))
        ]
        self.assertTrue(detect_conversion("任意对话"))


class TestScoring(unittest.TestCase):

    @patch("monitor.llm_ds")
    def test_score_returns_dict(self, mock_llm):
        from monitor import score_conversation
        mock_llm.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "score": 0.35,
                "problems": ["重复追问"],
                "bad_turn": "AI第3条",
                "suggestion": "补充FAQ"
            })))
        ]
        result = score_conversation("[顾客] 多少钱\n[AI] 留个微信\n[顾客] 你没回答")
        self.assertEqual(result["score"], 0.35)
        self.assertIn("重复追问", result["problems"])

    @patch("monitor.llm_ds")
    def test_score_fallback_on_bad_json(self, mock_llm):
        from monitor import score_conversation
        mock_llm.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="解析失败的文本"))
        ]
        result = score_conversation("任意对话")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["problems"], [])


class TestStats(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def test_append_and_load(self):
        import monitor as m
        from pathlib import Path
        orig_stats  = m.STATS
        orig_reports = m.REPORTS
        m.STATS   = Path(self.tmpdir) / "stats.json"
        m.REPORTS = Path(self.tmpdir)

        try:
            m.append_stats("2026-04-27", 83, 31, 11)
            data = m.load_stats()
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["date"], "2026-04-27")
            self.assertAlmostEqual(data[0]["rate"], 0.373, places=2)
            self.assertEqual(data[0]["bad"], 11)
        finally:
            m.STATS   = orig_stats
            m.REPORTS = orig_reports

    def test_append_overwrites_same_date(self):
        import monitor as m
        from pathlib import Path
        orig_stats  = m.STATS
        orig_reports = m.REPORTS
        m.STATS   = Path(self.tmpdir) / "stats.json"
        m.REPORTS = Path(self.tmpdir)
        try:
            m.append_stats("2026-04-27", 80, 28, 9)
            m.append_stats("2026-04-27", 83, 31, 11)
            data = m.load_stats()
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["total"], 83)
        finally:
            m.STATS   = orig_stats
            m.REPORTS = orig_reports


class TestDailyReport(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def _run_report(self, results):
        import monitor as m
        from pathlib import Path
        orig = m.REPORTS
        m.REPORTS = Path(self.tmpdir)
        try:
            return m.generate_daily_report("2026-04-27", results)
        finally:
            m.REPORTS = orig

    def test_report_contains_rate(self):
        results = [
            {"id": "abc123", "converted": True,  "score": {"score": 0.9, "problems": [], "bad_turn": "", "suggestion": ""}},
            {"id": "def456", "converted": False, "score": {"score": 0.3, "problems": ["重复追问"], "bad_turn": "AI回复", "suggestion": "补FAQ"}},
        ]
        path = self._run_report(results)
        content = path.read_text(encoding="utf-8")
        self.assertIn("50.0%", content)
        self.assertIn("重复追问", content)
        self.assertIn("补FAQ", content)

    def test_report_no_bad_section_when_all_good(self):
        results = [
            {"id": "abc123", "converted": True, "score": {"score": 0.9, "problems": [], "bad_turn": "", "suggestion": ""}},
        ]
        path = self._run_report(results)
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("劣质对话详情", content)


class TestWeeklyReport(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def test_weekly_report_shows_trend(self):
        import monitor as m
        import datetime
        from pathlib import Path
        orig_stats   = m.STATS
        orig_reports = m.REPORTS
        m.STATS   = Path(self.tmpdir) / "stats.json"
        m.REPORTS = Path(self.tmpdir)
        try:
            today = datetime.date.today()
            for i in range(7):
                day = today - datetime.timedelta(days=6 - i)
                m.append_stats(day.isoformat(), 80, 20 + i, 5)
            prev_monday = today - datetime.timedelta(days=today.weekday() + 7)
            for i in range(7):
                day = prev_monday + datetime.timedelta(days=i)
                m.append_stats(day.isoformat(), 80, 15, 8)

            path = m.generate_weekly_report()
            content = path.read_text(encoding="utf-8")
            self.assertIn("留资率趋势", content)
            self.assertIn("↑", content)
        finally:
            m.STATS   = orig_stats
            m.REPORTS = orig_reports


if __name__ == "__main__":
    unittest.main()
