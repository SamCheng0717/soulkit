# export.py
#
# 从 OceanBase/MySQL 导出聊天记录为 build.py 所需 CSV
# 用法:
#   python export.py --chatid-file raw/chatid.txt [--output raw/chatlogs.csv] [--limit N]
#   python export.py --hospital <医院名>           [--output raw/chatlogs.csv] [--limit N]
#
# 依赖: pip install pymysql python-dotenv
# 注意: 本文件中的 SQL 查询依赖特定数据库表结构，请根据自己的 schema 修改

import os, csv, argparse
from pathlib import Path
from dotenv import load_dotenv

try:
    import pymysql
except ImportError:
    raise SystemExit("缺少依赖: pip install pymysql")

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",   "localhost"),
    "port":     int(os.getenv("DB_PORT", "3306")),
    "user":     os.getenv("DB_USER",   ""),
    "password": os.getenv("DB_PASS",   ""),
    "database": os.getenv("DB_NAME",   ""),
    "charset":  "utf8mb4",
}

SQL = """
SELECT
    m.`to`          AS session_id,
    CASE WHEN u.user_type IN (1, 2) THEN '客服' ELSE '客户' END AS role,
    COALESCE(NULLIF(m.zh_content, ''), m.content)          AS message,
    m.timestamp
FROM hd_message m
JOIN hd_user u ON u.uid = m.`from`
JOIN (
    SELECT `to` as gid
    FROM hd_message
    WHERE type='group' AND sub_type='message'
      AND COALESCE(NULLIF(zh_content,''), content) NOT IN ('','custom')
    GROUP BY `to`
    HAVING COUNT(*) BETWEEN {min_msg} AND {max_msg}
    {scope_having}
) valid_groups ON valid_groups.gid = m.`to`
WHERE m.type    = 'group'
  AND m.sub_type = 'message'
  AND COALESCE(NULLIF(m.zh_content, ''), m.content) IS NOT NULL
  AND COALESCE(NULLIF(m.zh_content, ''), m.content) NOT IN ('', 'custom')
  {scope_where}
ORDER BY m.`to`, m.timestamp
{limit}
"""


def load_chatids(chatid_file: Path) -> list[int]:
    gids = []
    with open(chatid_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                gids.append(int(line))
    return gids


def export(output: Path, limit: int | None, hospital: str | None,
           chatid_file: Path | None, min_msg: int = 5, max_msg: int = 200):

    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:

            if chatid_file:
                gids = load_chatids(chatid_file)
                if not gids:
                    raise SystemExit(f"chatid 文件为空或格式不对: {chatid_file}")
                print(f"加载 chatid 文件: {len(gids)} 个群 ID")
                # gids 来自可信文件，全是整数，直接内嵌 IN 子句
                gid_list = ",".join(map(str, gids))
                scope_where  = f"AND m.`to` IN ({gid_list})"
                scope_having = f"AND `to` IN ({gid_list})"
                params = ()
                tag = f"[chatid:{len(gids)}groups] "

            elif hospital:
                scope_where  = "AND g.hospital_name = %s"
                scope_having = "AND `to` IN (SELECT gid FROM hd_groups WHERE hospital_name = %s)"
                params = (hospital, hospital)
                tag = f"[{hospital}] "

            else:
                scope_where  = ""
                scope_having = ""
                params = ()
                tag = ""

            limit_clause = f"LIMIT {limit}" if limit else ""

            sql = SQL.format(
                scope_where=scope_where,
                scope_having=scope_having,
                min_msg=min_msg,
                max_msg=max_msg,
                limit=limit_clause,
            )
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        print("查询结果为空，请检查过滤条件或数据库连接")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "role", "message"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "session_id": row["session_id"],
                "role":       row["role"],
                "message":    row["message"],
            })

    sessions = len({r["session_id"] for r in rows})
    print(f"导出完成: {tag}{len(rows)} 条消息 / {sessions} 个会话 -> {output}")
    print(f"下一步: python build.py {output}")


def main():
    parser = argparse.ArgumentParser(description="导出聊天记录为 cs-agent CSV")
    parser.add_argument("--chatid-file", "-f", default=None,              help="群 ID 文件路径（每行一个 gid），优先于 --hospital")
    parser.add_argument("--hospital",    "-H", default=None,              help="按 hospital_name 字段过滤（需根据自己的 schema 调整）")
    parser.add_argument("--output",      "-o", default="raw/chatlogs.csv",help="输出文件路径")
    parser.add_argument("--limit",       "-n", type=int, default=None,    help="最多导出 N 条消息（调试用）")
    parser.add_argument("--min-msg",           type=int, default=5,       help="会话最少消息数（默认5）")
    parser.add_argument("--max-msg",           type=int, default=200,     help="会话最多消息数（默认200，过滤超长群）")
    args = parser.parse_args()

    chatid_file = Path(args.chatid_file) if args.chatid_file else None
    export(Path(args.output), args.limit, args.hospital, chatid_file, args.min_msg, args.max_msg)


if __name__ == "__main__":
    main()
