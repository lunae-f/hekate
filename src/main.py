import sys
from pathlib import Path
# 親ディレクトリを sys.path に追加して 'src' パッケージのインポートを可能にする
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import datetime
import io
import os
import aiohttp
import discord
from discord import app_commands
from discord.ext import commands, tasks
import pytz
from google.genai import types

from src.config import config
from src.db import init_db, get_db_connection
from src.qmd_engine import QMDEngine
from src.memory_retriever import MemoryRetriever
from src.agent import AIAgent, AgentReply
from src.scheduler import TaskScheduler, tool_registry

import logging

# カスタムログフォーマッタの定義
class CustomFormatter(logging.Formatter):
    def format(self, record):
        level = f"[{record.levelname}]"
        # 警告レベルの表記ゆれを揃える (WARNING -> WARNING)
        levelname = record.levelname
        level = f"[{levelname}]"
        logger_name = f"[{record.name}]"
        return f"{level} {logger_name} {record.getMessage()}"

# ロギング設定
log_level = getattr(logging, config.log_level.upper(), logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CustomFormatter())

root_logger = logging.getLogger()
root_logger.setLevel(log_level)
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)
root_logger.addHandler(handler)

logger = logging.getLogger("main")
scheduler_logger = logging.getLogger("scheduler")
retriever_logger = logging.getLogger("retriever")


# Botの初期化 (Intentsの設定)
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)
timezone = pytz.timezone(config.timezone)

# 各種エンジンの初期化
qmd_engine = QMDEngine()
memory_retriever = MemoryRetriever(qmd_engine)
agent = AIAgent()
scheduler = TaskScheduler()

# デバウンス用バッファとタスク管理
# キー: channel_id, 値: { "messages": [message_objects], "task": asyncio.Task }
debounce_buffers = {}

# 定期タスクポーリング (1分ごと)
@tasks.loop(minutes=1.0)
async def poll_scheduled_tasks():
    scheduler_logger.debug("Checking for triggerable scheduled tasks...")
    try:
        triggerable_tasks = await scheduler.get_triggerable_tasks()
        for t in triggerable_tasks:
            task_id = t["id"]
            cron_expr = t["cron_expression"]
            channel_id = t["channel_id"]
            instruction = t["instruction"]
            tool_name = t["tool_name"]
            tool_args = t["tool_args"]

            channel = bot.get_channel(int(channel_id))
            if not channel:
                scheduler_logger.warning(f"Channel {channel_id} not found for task {task_id}.")
                await scheduler.reschedule_or_delete(task_id, cron_expr)
                continue

            if tool_name:
                scheduler_logger.info(f"Triggering scheduled task ID {task_id} in #{channel.name}. (Tool: '{tool_name}')")
                try:
                    await tool_registry.execute(tool_name, tool_args, bot, channel_id)
                except Exception as e:
                    scheduler_logger.error(f"Failed to execute tool {tool_name}: {e}")
                    await channel.send(f"⚠️ 定期タスク（ツール: {tool_name}）の実行中にエラーが発生しました。")
            
            elif instruction:
                scheduler_logger.info(f"Triggering scheduled task ID {task_id} in #{channel.name}. (Instruction: '{instruction}')")
                
                # タスク登録したユーザーのIDを取得
                task_user_id = t.get("user_id")
                
                context = await memory_retriever.get_context(instruction, [])
                try:
                    reply = await agent.generate_scheduled_reply(context, instruction)
                    
                    # ユーザーIDがある場合は自動メンション化
                    if task_user_id:
                        import re
                        mention_str = f"<@{task_user_id}>"
                        reply_stripped = reply.strip()
                        
                        # 文頭の平文メンション（例: @username または @ユーザー名）を <@ユーザーID> に置換
                        if reply_stripped.startswith("@"):
                            reply = re.sub(r'^@\w+', mention_str, reply_stripped, count=1)
                        else:
                            # メンションがなければ、メッセージの先頭にメンションを強制付与
                            reply = f"{mention_str}\n\n{reply}"
                            
                    await channel.send(reply)
                except Exception as e:
                    scheduler_logger.error(f"Failed to generate scheduled reply: {e}")
                    await channel.send("⚠️ 定期タスクのメッセージ生成中にエラーが発生しました。")

            await scheduler.reschedule_or_delete(task_id, cron_expr)

    except Exception as e:
        scheduler_logger.error(f"Exception in scheduler loop: {e}")

# 毎日深夜の「長期記憶整理」タスク (午前3時 JST)
@tasks.loop(hours=24.0)
async def daily_memory_consolidation():
    retriever_logger.info("Starting daily memory consolidation (consolidating past 24 hours of logs)...")
    try:
        # 過去24時間のログをまとめて要約し、long_term_memory.md に追記
        tz = pytz.timezone(config.timezone)
        yesterday = datetime.datetime.now(tz) - datetime.timedelta(days=1)
        yesterday_iso = yesterday.isoformat()

        async with get_db_connection() as conn:
            cursor = await conn.execute("""
            SELECT content, username, timestamp 
            FROM chat_history 
            WHERE timestamp >= ? 
            ORDER BY timestamp ASC
            """, (yesterday_iso,))
            rows = await cursor.fetchall()

        if not rows:
            retriever_logger.info("No chat logs to consolidate.")
            return

        # 会話テキストの構築
        chat_text = "\n".join([f"{r['timestamp']} - {r['username']}: {r['content']}" for r in rows])
        
        # 要約プロンプト
        prompt = f"""
以下は過去24時間のチャットログです。この中から、決定事項、重要な技術的議論、ユーザーに関する知識、解決されたエラー、エージェントへの要望などをコンパクトに要約してください。
今後の会話のコンテキストとして記憶しておくべき重要な情報のみに絞って箇年書きでまとめてください。

# チャットログ
{chat_text}
"""
        response = await agent.client.aio.models.generate_content(
            model=config.evaluator_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="あなたはチャットログを長期記憶用に要約する優秀な記録アシスタントです。",
                temperature=0.2,
            )
        )
        
        # long_term_memory.md にアペンド
        ROOT_DIR = Path(__file__).parent.parent
        LTM_PATH = ROOT_DIR / "data" / "memory" / "long_term_memory.md"
        LTM_PATH.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(LTM_PATH, mode="a", encoding="utf-8") as f:
            await f.write(f"\n### {yesterday.strftime('%Y-%m-%d')} の要約\n{response.text}\n")
            
        retriever_logger.info("Daily memory consolidation complete. Summary appended to long_term_memory.md.")

    except Exception as e:
        retriever_logger.error(f"Failed to consolidate daily memory: {e}")

@daily_memory_consolidation.before_loop
async def before_daily_memory_consolidation():
    # 毎日午前3時（JST）に動作するように、最初の実行までのディレイを計算して待つ
    await bot.wait_until_ready()
    tz = pytz.timezone(config.timezone)
    now = datetime.datetime.now(tz)
    
    # 次の午前3時
    target_time = now.replace(hour=3, minute=0, second=0, microsecond=0)
    if now >= target_time:
        target_time += datetime.timedelta(days=1)
        
    delay_seconds = (target_time - now).total_seconds()
    retriever_logger.info(f"Scheduled to run daily memory consolidation JST 03:00. Waiting {delay_seconds} seconds...")
    await asyncio.sleep(delay_seconds)

# キャンセルボタン付きViewの定義
class CancelView(discord.ui.View):
    def __init__(self, task: asyncio.Task, status_msg: discord.Message):
        super().__init__(timeout=60.0)
        self.task = task
        self.status_msg = status_msg

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.danger, emoji="🛑")
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.warning(f"Message processing for channel {interaction.channel_id} cancelled by user.")
        # タスクをキャンセル
        self.task.cancel()
        self.disable_all_items()
        
        # UIの変更
        embed = discord.Embed(
            title="Cancelled 🛑",
            description="処理がユーザーによって中断されました。",
            color=0xe74c3c
        )
        try:
            await interaction.response.edit_message(embed=embed, view=None)
        except Exception:
            pass

        # 3秒後にステータスメッセージを削除
        await asyncio.sleep(3)
        try:
            await self.status_msg.delete()
        except discord.errors.NotFound:
            pass
        except Exception as e:
            logger.error(f"Failed to delete status message: {e}")

# スラッシュコマンドの実装
@bot.tree.command(name="schedule_list", description="登録されている定期タスクの一覧を表示します。")
async def schedule_list(interaction: discord.Interaction):
    logger.info(f"User {interaction.user.display_name} executed slash command /schedule_list")
    await interaction.response.defer()
    try:
        tasks_list = await scheduler.list_tasks()
        if not tasks_list:
            await interaction.followup.send("現在登録されているスケジュールタスクはありません。")
            return

        embed = discord.Embed(
            title="📅 スケジュールタスク一覧",
            color=0x3498db,
            timestamp=datetime.datetime.now(timezone)
        )
        for t in tasks_list:
            schedule_str = f"cron: `{t['cron_expression']}`" if t["cron_expression"] else f"日時: `{t['run_at']}`"
            action_str = f"指示: {t['instruction']}" if t["instruction"] else f"ツール: {t['tool_name']}"
            embed.add_field(
                name=f"ID: {t['id']} | チャンネル: <#{t['channel_id']}>",
                value=f"{schedule_str}\n{action_str}\n次回実行: `{t['next_run']}`",
                inline=False
            )
        await interaction.followup.send(embed=embed)
    except Exception as e:
        await interaction.followup.send(f"❌ タスク一覧の取得中にエラーが発生しました: {e}")

@bot.tree.command(name="schedule_delete", description="指定したIDの定期タスクを削除します。")
@app_commands.describe(task_id="削除するタスクのID")
async def schedule_delete(interaction: discord.Interaction, task_id: int):
    logger.info(f"User {interaction.user.display_name} executed slash command /schedule_delete (Parameters: task_id={task_id})")
    await interaction.response.defer()
    try:
        success = await scheduler.delete_task(task_id)
        if success:
            await interaction.followup.send(f"✅ 定期タスク (ID: {task_id}) を削除しました。")
        else:
            await interaction.followup.send(f"❌ 指定されたID (ID: {task_id}) のタスクが見つかりませんでした。")
    except Exception as e:
        await interaction.followup.send(f"❌ タスクの削除中にエラーが発生しました: {e}")

@bot.tree.command(name="schedule_add", description="定期タスクを手動で登録します。")
@app_commands.describe(
    cron="cron式 (例: '0 9 * * 1' 毎週月曜9時)",
    instruction="実行させる自然言語指示 (例: '今日の天気を要約して')"
)
async def schedule_add(interaction: discord.Interaction, cron: str, instruction: str):
    logger.info(f"User {interaction.user.display_name} executed slash command /schedule_add (Parameters: cron={cron}, instruction={instruction})")
    await interaction.response.defer()
    try:
        # スケジュール計算テスト
        scheduler.calculate_next_run(cron_expression=cron)
        
        task_id = await scheduler.add_task(
            channel_id=str(interaction.channel_id),
            user_id=str(interaction.user.id),
            cron_expression=cron,
            instruction=instruction
        )
        await interaction.followup.send(
            f"✅ 定期タスク (ID: {task_id}) を登録しました。\n"
            f"スケジュール: `{cron}`\n"
            f"指示: {instruction}"
        )
    except Exception as e:
        await interaction.followup.send(f"❌ スケジュール登録に失敗しました。cron式が正しいかご確認ください: {e}")

@bot.tree.command(name="agent_status", description="Botの動作ステータスを確認します。")
async def agent_status(interaction: discord.Interaction):
    logger.info(f"User {interaction.user.display_name} executed slash command /agent_status")
    await interaction.response.defer()
    try:
        tasks_list = await scheduler.list_tasks()
        
        async with get_db_connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) as count FROM chat_history")
            log_count = (await cursor.fetchone())["count"]

        embed = discord.Embed(title="🤖 エージェント動作ステータス", color=0x2ecc71)
        embed.add_field(name="タイムゾーン", value=config.timezone, inline=True)
        embed.add_field(name="判定モデル", value=config.evaluator_model, inline=True)
        embed.add_field(name="返答モデル", value=config.generator_model, inline=True)
        embed.add_field(name="登録スケジュール数", value=str(len(tasks_list)), inline=True)
        embed.add_field(name="キャッシュされたログ件数", value=f"{log_count} 件", inline=True)
        embed.add_field(name="トリガーキーワード数", value=f"{len(config.trigger_keywords)} 個", inline=True)
        
        await interaction.followup.send(embed=embed)
    except Exception as e:
        await interaction.followup.send(f"❌ ステータス取得に失敗しました: {e}")

# メッセージごとのメイン非同期処理
async def process_debounce_messages(channel_id: int):
    # バッファからメッセージを回収してクリア
    buffer = debounce_buffers.get(channel_id)
    if not buffer:
        return
        
    messages = buffer["messages"]
    # タイムスタンプ順にソート
    messages.sort(key=lambda m: m.created_at)
    
    # バッファのクリア
    if channel_id in debounce_buffers:
        del debounce_buffers[channel_id]

    primary_msg = messages[-1] # 代表メッセージ (最新の発言)
    channel = primary_msg.channel
    channel_name = f"#{channel.name}" if hasattr(channel, "name") else str(channel_id)

    logger.info(f"Debounce finished. Merged {len(messages)} messages in channel {channel_name} for processing.")

    # 1. 添付ファイルの取得とQMDインジェスト
    # 連投されたメッセージ全体のテキストと添付ファイルを統合する
    full_content_lines = []
    all_attachments = []

    for msg in messages:
        # テキストの結合
        user_prefix = f"{msg.author.display_name}: "
        full_content_lines.append(user_prefix + msg.clean_content)

        # 添付ファイルの非同期読み込み
        for att in msg.attachments:
            att_data = {"filename": att.filename, "content": None}
            # テキストファイルの場合、中身をログ化するためにデコードして読込
            if att.content_type and (att.content_type.startswith("text/") or att.filename.endswith((".py", ".json", ".toml", ".yaml", ".ini", ".md", ".js", ".ts"))):
                try:
                    bytes_data = await att.read()
                    att_data["content"] = bytes_data.decode("utf-8", errors="ignore")
                except Exception as e:
                    logger.warning(f"Failed to read text attachment {att.filename}: {e}")
            all_attachments.append(att_data)

    # インジェスト用の結合テキスト
    ingest_content = "\n".join(full_content_lines)

    # 過去ログへのインジェスト実行 (QMD Markdown, chat_history, embeddings)
    # タイムスタンプは最新メッセージのものをJST/現地時間に変換
    timestamp_local = primary_msg.created_at.astimezone(timezone)
    await qmd_engine.ingest_message(
        message_id=str(primary_msg.id),
        timestamp=timestamp_local,
        channel_id=str(channel.id),
        channel_name=channel.name if not isinstance(channel, discord.Thread) else channel.parent.name,
        user_id=str(primary_msg.author.id),
        username=primary_msg.author.display_name,
        content=ingest_content,
        attachments=all_attachments
    )

    # 2. トリガー判定 (Pre-Filter)
    # A) 自分宛てのメンション、または自分への返信
    has_mention = bot.user.mentioned_in(primary_msg) or (
        primary_msg.reference and 
        primary_msg.reference.cached_message and 
        primary_msg.reference.cached_message.author == bot.user
    )

    # B) スレッド内であり、Botがスレッドに参加しているか
    is_joined_thread = isinstance(channel, discord.Thread) and channel.me is not None

    # C) 添付ファイルの存在
    has_attachments = len(all_attachments) > 0

    # D) トリガーキーワードの検出 (結合された内容で部分一致チェック)
    matched_keyword = None
    for kw in config.trigger_keywords:
        if kw in ingest_content:
            matched_keyword = kw
            break
    has_trigger_keyword = matched_keyword is not None

    # トリガーの決定
    is_triggered = has_mention or is_joined_thread or has_attachments or has_trigger_keyword

    if not is_triggered:
        # トリガーされない場合はここで終了 (ログ保存のみ完了した状態)
        logger.debug(f"Message {primary_msg.id} skipped: No mention, no trigger keywords, and not in active thread.")
        return

    # トリガー理由の構成
    reasons = []
    if has_mention:
        reasons.append("メンション検知")
    if has_trigger_keyword:
        reasons.append(f"キーワード '{matched_keyword}' を検出")
    if is_joined_thread:
        reasons.append("アクティブスレッド")
    if has_attachments:
        reasons.append("添付ファイル検知")
    
    reason_str = " / ".join(reasons)
    logger.info(f"Message {primary_msg.id} triggered LLM evaluation. (Reason: {reason_str})")

    # 3. LLM 判定と応答生成の非同期実行
    # タスクがキャンセルされた場合に安全に後片付けできるよう、全体の処理をタスク化する
    async def run_llm_chain():
        status_msg = None
        try:
            # A) 判定中ステータス Embed を送信 (中断用ボタン付きView)
            embed = discord.Embed(
                title="Thinking... 🤔",
                description="メッセージを分析しています...",
                color=0x3498db
            )
            # Viewに自身を実行しているTaskを渡すことでキャンセル可能にする
            current_task = asyncio.current_task()
            status_msg = await channel.send(embed=embed)
            view = CancelView(current_task, status_msg)
            await status_msg.edit(view=view)

            # 過去のコンテキスト(QMD想起 + 長期記憶)を検索
            context = await memory_retriever.get_context(primary_msg.clean_content, [
                {"message_id": str(m.id), "username": m.author.display_name, "content": m.clean_content} for m in messages
            ])

            # 直近会話履歴（短期記憶）の文字列整形
            recent_history_lines = []
            for msg in messages[:-1]: # 最新メッセージ以外
                recent_history_lines.append(f"{msg.author.display_name}: {msg.clean_content}")
            recent_history = "\n".join(recent_history_lines)

            # 画像添付ファイルの Part 作成
            image_parts = []
            for msg in messages:
                for att in msg.attachments:
                    if att.content_type and att.content_type.startswith("image/"):
                        try:
                            img_bytes = await att.read()
                            part = types.Part.from_bytes(
                                data=img_bytes,
                                mime_type=att.content_type
                            )
                            image_parts.append(part)
                        except Exception as e:
                            logger.warning(f"Failed to load image {att.filename}: {e}")

            # 最新メッセージにテキスト添付ファイルの内容を結合
            current_message_with_attachments = primary_msg.clean_content
            for att in all_attachments:
                if att["content"] is not None:
                    current_message_with_attachments += f"\n\n--- 添付ファイル: {att['filename']} ---\n{att['content']}"

            # B) Evaluator による応答判定と一次回答生成
            now_iso = datetime.datetime.now(timezone).isoformat()
            decision = None
            triggered_fallback = False

            try:
                decision = await agent.evaluate_and_reply(
                    context,
                    recent_history,
                    current_message_with_attachments,
                    str(primary_msg.id),
                    now_iso,
                    image_parts
                )
            except Exception as e:
                # パースエラー等の例外が発生した場合は、Fallback（エスカレーション）を強制する
                logger.warning(f"Fallback triggered by evaluate_and_reply execution/parse error: {e}")
                triggered_fallback = True

            # 例外が発生しなかった場合
            if decision is not None:
                # スケジュール指示がある場合は登録
                if decision.new_schedule:
                    ns = decision.new_schedule
                    try:
                        task_id = await scheduler.add_task(
                            channel_id=str(channel.id),
                            user_id=str(primary_msg.author.id),
                            cron_expression=ns.cron_expression,
                            run_at=ns.run_at,
                            instruction=ns.instruction,
                            tool_name=ns.tool_name,
                            tool_args=ns.tool_args
                        )
                        schedule_desc = f"cron: `{ns.cron_expression}`" if ns.cron_expression else f"日時: `{ns.run_at}`"
                        await channel.send(f"✅ スケジュール登録しました (ID: {task_id})。 {schedule_desc}")
                    except Exception as e:
                        scheduler_logger.error(f"Failed to auto-schedule task: {e}")
                        await channel.send("⚠️ スケジュールタスクの自動登録に失敗しました。")

                # 応答が不要な場合はステータス表示を削除して終了
                if not decision.should_respond:
                    try:
                        await status_msg.delete()
                    except discord.errors.NotFound:
                        pass
                    return

                # 自己評価による Fallback チェック
                if decision.requires_escalation or decision.confidence_score <= 3:
                    triggered_fallback = True
                    logger.info(f"Fallback triggered by self-evaluation: confidence_score={decision.confidence_score}, requires_escalation={decision.requires_escalation}")

            # C) 応答を返す場合は「返答作成中」に Embed を更新
            embed.title = "Generating reply... ✍️"
            embed.description = "返答を作成しています..."
            try:
                await status_msg.edit(embed=embed)
            except discord.errors.NotFound:
                pass

            reply = None
            # Fallback (エスカレーション) の実行
            if triggered_fallback:
                logger.info(f"Escalating query response generation to premium model ({config.model_premium})...")
                # UXの向上: 進捗 Embed を「良い感じに」更新
                embed.title = "Re-evaluating Response... 🚀"
                embed.description = "回答を再評価し、高性能モデルで再生成しています..."
                embed.color = 0xe67e22 # オレンジ色の警告/通知カラー
                try:
                    await status_msg.edit(embed=embed)
                except discord.errors.NotFound:
                    pass
                
                # premium モデルで再生成
                reply = await agent.generate_reply(
                    context, 
                    recent_history, 
                    current_message_with_attachments,
                    str(primary_msg.id), 
                    config.model_premium, 
                    image_parts
                )
            else:
                # 1回目の判定時に生成された回答をそのまま使用
                reply = AgentReply(
                    reply_content=decision.reply_content,
                    attachment_content=decision.attachment_content,
                    attachment_filename=decision.attachment_filename
                )

            # 送信処理
            # 1. 添付ファイルがある場合
            file_to_send = None
            if reply.attachment_content and reply.attachment_filename:
                # メモリ上でバイナリファイルを作成して添付
                file_io = io.BytesIO(reply.attachment_content.encode("utf-8"))
                file_to_send = discord.File(file_io, filename=reply.attachment_filename)
            
            # 2. 返答テキストがDiscord制限を超える場合は自動ファイル化
            main_content = reply.reply_content
            if len(main_content) > 2000:
                file_io = io.BytesIO(main_content.encode("utf-8"))
                file_to_send = discord.File(file_io, filename="reply_long.txt")
                main_content = "⚠️ 返答が長文（2,000文字超）のため、テキストファイルに変換して添付しました。"

            # 送信
            if file_to_send:
                await channel.send(content=main_content, file=file_to_send)
            else:
                await channel.send(content=main_content)

            # D) 送信完了後、ステータスメッセージを削除
            try:
                await status_msg.delete()
            except discord.errors.NotFound:
                pass

        except asyncio.CancelledError:
            # 処理が中断された場合
            logger.info(f"Message processing task for channel {channel_id} was cancelled.")
            # 中断時のクリーンアップ
            if status_msg:
                try:
                    # すでに View 側で Cancelled 🛑 に更新されているため、ステータスメッセージの自動削除のみ行う
                    await asyncio.sleep(2)
                    await status_msg.delete()
                except discord.errors.NotFound:
                    pass
                except Exception as e:
                    logger.error(f"Failed to delete status message on cancel: {e}")
        except Exception as e:
            logger.error(f"Exception in LLM execution: {e}")
            if status_msg:
                try:
                    embed_err = discord.Embed(title="Error ❌", description=f"処理中にエラーが発生しました: {e}", color=0xe74c3c)
                    await status_msg.edit(embed=embed_err, view=None)
                    await asyncio.sleep(5)
                    await status_msg.delete()
                except discord.errors.NotFound:
                    pass

    # メッセージ処理をタスクとしてバックグラウンドで開始
    asyncio.create_task(run_llm_chain())

# メッセージ受信イベント
@bot.event
async def on_message(message: discord.Message):
    # 自分自身の発言、または他のBotの発言は即スルー
    if message.author == bot.user or message.author.bot:
        return

    channel_id = message.channel.id
    channel = message.channel
    channel_name = f"#{channel.name}" if hasattr(channel, "name") else str(channel_id)
    
    # メッセージ検知ログ
    logger.info(f"Received message from {message.author.display_name} in {channel_name} (メッセージID: {message.id})")

    # メッセージ処理用のバッファを生成・更新
    is_new_debounce = channel_id not in debounce_buffers
    if is_new_debounce:
        debounce_buffers[channel_id] = {"messages": [], "task": None}
        # デバウンス待機開始ログ
        logger.debug(f"Debounce started for channel {channel_id}. Waiting 1.5s for successive messages...")

    buffer = debounce_buffers[channel_id]
    buffer["messages"].append(message)

    # 既存のデバウンス待ちタスクがあればキャンセル
    if buffer["task"] and not buffer["task"].done():
        buffer["task"].cancel()

    # 1.5秒待ってから処理を実行するデバウンスタスクを作成
    async def debounce_wait():
        try:
            await asyncio.sleep(1.5)
            # 待機が終わったら結合処理を起動
            await process_debounce_messages(channel_id)
        except asyncio.CancelledError:
            # キャンセルされた場合は何もしない (後続の連投にマージされる)
            pass

    buffer["task"] = asyncio.create_task(debounce_wait())

# グローバルな初期化フラグ
is_ready_initialized = False

# 起動完了イベント
@bot.event
async def on_ready():
    global is_ready_initialized
    logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")
    
    if is_ready_initialized:
        logger.info("Bot reconnected. Skipping database and task initialization.")
        return

    # データベースの初期化
    await init_db()
    logger.info("Database initialization complete.")

    # 定期タスクループの開始
    if not poll_scheduled_tasks.is_running():
        poll_scheduled_tasks.start()
    if not daily_memory_consolidation.is_running():
        daily_memory_consolidation.start()
    logger.info("Background periodic tasks started.")

    # スラッシュコマンドの登録 (Tree同期)
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} slash commands.")
    except Exception as e:
        logger.error(f"Failed to sync slash commands: {e}")

    is_ready_initialized = True

# エントリポイント
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
