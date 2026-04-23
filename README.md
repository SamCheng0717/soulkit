# cs-agent

传入聊天记录，得到一个克隆原客服风格的 AI Agent。

- **知识库**：从对话中提取 FAQ，存为可读的 Markdown 文件
- **人格**：蒸馏出客服的沟通风格和决策逻辑，生成 SOUL.md
- **部署**：一键同步到 Dify，接入对话系统

## 快速开始

```bash
git clone https://github.com/you/cs-agent
cd cs-agent
pip install -r requirements.txt
cp .env.example .env   # 填入 API Key
```

## 使用

```bash
# 方式 A：从数据库直接导出（OceanBase / MySQL）
python export.py                        # 默认输出 raw/chatlogs.csv
python export.py --limit 500            # 调试时只导出 500 条

# 方式 B：手动准备 CSV，放进 raw/
cp your_chatlogs.csv raw/

# 生成知识库 + 人格（约 5-15 分钟，取决于数据量）
python build.py raw/chatlogs.csv

# 推送到 Dify
python sync.py
```

## 输出文件

| 路径 | 内容 |
|------|------|
| `wiki/faq/*.md` | 从对话提取的 FAQ 知识页面 |
| `prompts/SOUL.md` | 蒸馏出的客服人格与决策风格 |
| `wiki/index.md` | 知识库全局索引 |
| `wiki/log.md` | 操作记录 |

## CSV 格式要求

文件必须包含以下三列：

```
session_id,role,message
001,客服,您好请问有什么可以帮您
001,客户,我的订单什么时候发货
001,客服,您好，一般3-5个工作日发货
002,客户,退货怎么申请
002,客服,在订单详情页点击申请退货即可
```

- `session_id`：同一次对话用同一个 ID
- `role`：`客服` 或 `客户`（也支持 `agent`/`user`）
- `message`：消息内容

## 接入 Dify

运行 `sync.py` 后，在 Dify 中：

1. **知识检索节点** → 选择同步进来的知识库
2. **LLM 节点 System Prompt** → 粘贴 `prompts/SOUL.md` 的全部内容

## 自定义编译规范

修改 `wiki/schema.md` 可以调整 AI 提取 FAQ 的标准和格式。  
首次运行会自动从 `templates/schema.md` 复制一份默认规范。

## 配置项（.env）

```
DEEPSEEK_API_KEY=sk-        # DeepSeek API Key
DEEPSEEK_BASE_URL=          # 默认 https://api.deepseek.com
DIFY_API_KEY=app-           # Dify 应用 API Key  
DIFY_BASE_URL=              # Dify 地址，如 http://localhost:80
DIFY_DATASET_ID=            # 在 Dify 创建知识库后填入
```
