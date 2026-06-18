import asyncio
import aiosqlite
import logging
from google import genai
from src.config import config
from src.db import init_db, DB_PATH
from src.qmd_engine import QMDEngine
from src.memory_retriever import MemoryRetriever
from src.agent import AIAgent
from src.scheduler import TaskScheduler

logger = logging.getLogger("context")

class AgentContext:
    """アプリケーション全体で共有するデータベース接続や各種エンジンのライフサイクルを管理するコンテキストクラス"""
    def __init__(self):
        self.db_conn = None
        self.gemini_client = None
        
        self.qmd_engine = None
        self.memory_retriever = None
        self.agent = None
        self.scheduler = None
        self.persona_manager = None

    async def initialize(self):
        """データベース接続と各種エンジンを非同期で初期化する"""
        logger.info("Initializing AgentContext...")
        
        # 1. データベースの初期化
        await init_db()
        
        # 2. 持続的な SQLite 接続を開く
        self.db_conn = await aiosqlite.connect(DB_PATH)
        await self.db_conn.execute("PRAGMA foreign_keys = ON;")
        self.db_conn.row_factory = aiosqlite.Row
        
        # 3. Gemini SDK クライアントの作成
        self.gemini_client = genai.Client()
        
        # 4. 各種エンジンを依存関係注入(DI)でインスタンス化
        self.qmd_engine = QMDEngine(self.db_conn, self.gemini_client)
        self.memory_retriever = MemoryRetriever(self.qmd_engine)
        
        from src.persona_manager import PersonaManager
        self.persona_manager = PersonaManager()
        self.agent = AIAgent(self.gemini_client, self.persona_manager)
        self.scheduler = TaskScheduler(self.db_conn)
        
        logger.info("AgentContext initialization complete.")

    async def close(self):
        """各種リソースを安全に解放し、仕掛かり中のバックグラウンドタスクを完了させる（Graceful Shutdown）"""
        logger.info("Closing AgentContext...")
        
        if self.qmd_engine:
            await self.qmd_engine.wait_for_tasks()
            
        if self.db_conn:
            await self.db_conn.close()
            logger.info("Database connection closed.")
            
        if self.gemini_client:
            try:
                if hasattr(self.gemini_client, "close") and asyncio.iscoroutinefunction(self.gemini_client.close):
                    await self.gemini_client.close()
                elif hasattr(self.gemini_client, "aio") and hasattr(self.gemini_client.aio, "close"):
                    # aio.close がコルーチンである場合に await する
                    close_func = self.gemini_client.aio.close
                    if asyncio.iscoroutinefunction(close_func):
                        await close_func()
                    else:
                        close_func()
                logger.info("Gemini client session closed.")
            except Exception as e:
                logger.warning(f"Error closing Gemini client: {e}")
                
        logger.info("AgentContext closed successfully.")
