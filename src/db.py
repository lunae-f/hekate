import os
import contextlib
import aiosqlite
from pathlib import Path

# DBパスの設定
ROOT_DIR = Path(__file__).parent.parent
DB_DIR = ROOT_DIR / "data" / "index"
DB_PATH = DB_DIR / "agent.db"
ATTACHMENTS_DIR = ROOT_DIR / "data" / "attachments"

async def init_db():
    # 保存ディレクトリの作成
    DB_DIR.mkdir(parents=True, exist_ok=True)
    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(DB_PATH) as db:
        # SQLiteの外部キー制約を明示的に有効化
        await db.execute("PRAGMA foreign_keys = ON;")
        
        # 1. chat_history テーブルの作成 (QMDキャッシュ & トークンキャッシュ)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            message_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            channel_name TEXT NOT NULL,
            user_id TEXT NOT NULL,
            username TEXT NOT NULL,
            content TEXT NOT NULL,
            tokens TEXT NOT NULL,
            attachments TEXT NOT NULL
        );
        """)
        
        # 2. embeddings テーブルの作成 (埋め込みベクトルBLOB)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            message_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            FOREIGN KEY (message_id) REFERENCES chat_history (message_id) ON DELETE CASCADE
        );
        """)
        
        # 3. scheduled_tasks テーブルの作成 (cronおよび予約タスク)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cron_expression TEXT,
            run_at TEXT,
            instruction TEXT,
            tool_name TEXT,
            tool_args TEXT,
            channel_id TEXT NOT NULL,
            next_run TEXT NOT NULL,
            user_id TEXT
        );
        """)
        
        # 4. settings テーブルの作成 (グローバル設定)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)

        # 5. ignored_channels テーブルの作成 (無視するチャンネルリスト)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS ignored_channels (
            channel_id TEXT PRIMARY KEY,
            channel_name TEXT NOT NULL
        );
        """)
        
        # 既存データベース向けの自動マイグレーション (user_id カラムの追加)
        async with db.execute("PRAGMA table_info(scheduled_tasks);") as cursor:
            columns = [row[1] for row in await cursor.fetchall()]
        if "user_id" not in columns:
            await db.execute("ALTER TABLE scheduled_tasks ADD COLUMN user_id TEXT;")
            
        await db.commit()

@contextlib.asynccontextmanager
async def get_db_connection():
    db = await aiosqlite.connect(DB_PATH)
    try:
        await db.execute("PRAGMA foreign_keys = ON;")
        # 辞書ライクにレコードアクセスできるようにローファクトリを設定
        db.row_factory = aiosqlite.Row
        yield db
    finally:
        await db.close()
