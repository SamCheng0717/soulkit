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


if __name__ == "__main__":
    unittest.main()
