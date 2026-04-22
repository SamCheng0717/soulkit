# sync.py
#
# wiki/ → Dify 知识库（增量同步，幂等）
# 用法: python sync.py

import os, json, hashlib, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE    = os.getenv("DIFY_BASE_URL", "http://localhost:80") + "/v1"
HEADERS = {"Authorization": f"Bearer {os.getenv('DIFY_API_KEY', '')}"}
DS_ID   = os.getenv("DIFY_DATASET_ID", "")
STATE   = Path(".sync_state.json")
SKIP    = {"schema.md", "index.md", "log.md"}

def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}

def save_state(s: dict):
    STATE.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")

def push(name: str, content: str):
    r = requests.post(
        f"{BASE}/datasets/{DS_ID}/document/create_by_text",
        headers=HEADERS,
        json={
            "name": name,
            "text": content,
            "indexing_technique": "high_quality",
            "process_rule": {"mode": "automatic"},
        },
        timeout=30,
    )
    r.raise_for_status()

def main():
    if not DS_ID:
        print("✗ DIFY_DATASET_ID 未配置，请先在 Dify 创建知识库并填入 .env")
        return

    state   = load_state()
    changed = 0

    for md in sorted(Path("wiki").rglob("*.md")):
        if md.name in SKIP:
            continue
        content = md.read_text(encoding="utf-8")
        h       = hashlib.md5(content.encode()).hexdigest()
        key     = str(md)

        if state.get(key) == h:
            continue

        try:
            push(md.stem, content)
            state[key] = h
            changed += 1
            print(f"  ↑ {md.stem}")
        except Exception as e:
            print(f"  ✗ {md.stem}: {e}")

    save_state(state)
    print(f"\n同步完成，推送 {changed} 个变更")

if __name__ == "__main__":
    main()
