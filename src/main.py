import sys
from pathlib import Path
# 親ディレクトリを sys.path に追加して 'src' パッケージのインポートを可能にする
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import os
import discord
from discord.ext import commands

from src.logger import setup_logging
from src.context import AgentContext
from src.cogs.scheduler_cog import setup as setup_scheduler
from src.cogs.agent_cog import setup as setup_agent

import logging
logger = logging.getLogger("main")

# ロギングの初期化 (logger.pyより)
setup_logging()

# Botの初期化 (Intentsの設定)
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

# アプリケーションコンテキストのインスタンス化
context = AgentContext()

# グローバルな初期化フラグ
is_ready_initialized = False

@bot.event
async def on_ready():
    global is_ready_initialized
    logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")
    
    if is_ready_initialized:
        logger.info("Bot reconnected. Skipping database and task initialization.")
        return

    # 共通コンテキスト（DB、クライアント、各種エンジン）の非同期初期化
    await context.initialize()

    # Cogs の登録と初期化 (依存注入)
    await setup_scheduler(bot, context)
    await setup_agent(bot, context)
    logger.info("Discord Cogs registered successfully.")

    # スラッシュコマンドの登録 (Tree同期)
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} slash commands.")
    except Exception as e:
        logger.error(f"Failed to sync slash commands: {e}")

    is_ready_initialized = True

# closeメソッドをオーバーライドして、Bot停止時にコンテキストをクローズさせる (Graceful Shutdown)
original_close = bot.close

async def close():
    logger.info("Bot close process triggered. Releasing resources...")
    await context.close()
    await original_close()

bot.close = close

def main():
    token = os.environ.get("DISCORD_TOKEN")
    if not token or token == "your_discord_bot_token_here":
        logger.error("[Error] DISCORD_TOKEN is not set or not configured.")
        return
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.error("[Error] GEMINI_API_KEY is not set or not configured.")
        return

    bot.run(token)

if __name__ == "__main__":
    main()
