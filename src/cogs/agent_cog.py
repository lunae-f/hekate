import datetime
import logging
import asyncio
import io
import os
import aiofiles
from pathlib import Path
import discord
from discord import app_commands
from discord.ext import commands, tasks
import pytz
from google.genai import types

from src.config import config
from src.agent import AgentReply
from src.ui import CancelView, create_status_embed

logger = logging.getLogger("agent_cog")

class AgentCog(commands.Cog):
    """エージェントメッセージ処理、デバウンス、想起、一次判定・応答生成、長期記憶要約バッチを管理する Cog"""
    def __init__(self, bot, context):
        self.bot = bot
        self.context = context
        self.timezone = pytz.timezone(config.timezone)
        self.debounce_buffers = {}

    async def cog_load(self):
        """Cogロード時に深夜メモリ要約を開始する"""
        if not self.daily_memory_consolidation.is_running():
            self.daily_memory_consolidation.start()
        logger.info("Daily memory consolidation task started.")

    async def cog_unload(self):
        """Cogアンロード時に深夜メモリ要約を停止する"""
        self.daily_memory_consolidation.cancel()
        logger.info("Daily memory consolidation task stopped.")

    @tasks.loop(hours=24.0)
    async def daily_memory_consolidation(self):
        logger.info("Starting daily memory consolidation (consolidating past 24 hours of logs)...")
        try:
            yesterday = datetime.datetime.now(self.timezone) - datetime.timedelta(days=1)
            yesterday_iso = yesterday.isoformat()

            cursor = await self.context.db_conn.execute("""
            SELECT content, username, timestamp 
            FROM chat_history 
            WHERE timestamp >= ? 
            ORDER BY timestamp ASC
            """, (yesterday_iso,))
            rows = await cursor.fetchall()

            if not rows:
                logger.info("No chat logs to consolidate.")
                return

            chat_text = "\n".join([f"{r['timestamp']} - {r['username']}: {r['content']}" for r in rows])
            
            prompt = f"""
以下は過去24時間のチャットログです。この中から、決定事項、重要な技術的議論、ユーザーに関する知識、解決されたエラー、エージェントへの要望などをコンパクトに要約してください。
今後の会話のコンテキストとして記憶しておくべき重要な情報のみに絞って箇条書きでまとめてください。

# チャットログ
{chat_text}
"""
            # AIAgent経由で要約メッセージを生成
            response_text = await self.context.agent.generate_scheduled_reply("", prompt)
            
            LTM_PATH = Path(__file__).parent.parent.parent / "data" / "memory" / "long_term_memory.md"
            LTM_PATH.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(LTM_PATH, mode="a", encoding="utf-8") as f:
                await f.write(f"\n### {yesterday.strftime('%Y-%m-%d')} の要約\n{response_text}\n")
                
            logger.info("Daily memory consolidation complete. Summary appended to long_term_memory.md.")

        except Exception as e:
            logger.error(f"Failed to consolidate daily memory: {e}")

    @daily_memory_consolidation.before_loop
    async def before_daily_memory_consolidation(self):
        await self.bot.wait_until_ready()
        now = datetime.datetime.now(self.timezone)
        target_time = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if now >= target_time:
            target_time += datetime.timedelta(days=1)
            
        delay_seconds = (target_time - now).total_seconds()
        logger.info(f"Scheduled to run daily memory consolidation JST 03:00. Waiting {delay_seconds} seconds...")
        await asyncio.sleep(delay_seconds)

    @app_commands.command(name="agent_status", description="Botの動作ステータスを確認します。")
    async def agent_status(self, interaction: discord.Interaction):
        logger.info(f"User {interaction.user.display_name} executed slash command /agent_status")
        await interaction.response.defer()
        try:
            tasks_list = await self.context.scheduler.list_tasks()
            cursor = await self.context.db_conn.execute("SELECT COUNT(*) as count FROM chat_history")
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

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author == self.bot.user or message.author.bot:
            return

        channel_id = message.channel.id
        channel_name = f"#{message.channel.name}" if hasattr(message.channel, "name") else str(channel_id)
        logger.info(f"Received message from {message.author.display_name} in {channel_name} (ID: {message.id})")

        is_new_debounce = channel_id not in self.debounce_buffers
        if is_new_debounce:
            self.debounce_buffers[channel_id] = {"messages": [], "task": None}
            logger.debug(f"Debounce started for channel {channel_id}. Waiting 1.5s for successive messages...")

        buffer = self.debounce_buffers[channel_id]
        buffer["messages"].append(message)

        if buffer["task"] and not buffer["task"].done():
            buffer["task"].cancel()

        async def debounce_wait():
            try:
                await asyncio.sleep(1.5)
                await self.process_debounce_messages(channel_id)
            except asyncio.CancelledError:
                pass

        buffer["task"] = asyncio.create_task(debounce_wait())

    async def process_debounce_messages(self, channel_id: int):
        buffer = self.debounce_buffers.get(channel_id)
        if not buffer:
            return
            
        messages = buffer["messages"]
        messages.sort(key=lambda m: m.created_at)
        
        if channel_id in self.debounce_buffers:
            del self.debounce_buffers[channel_id]

        primary_msg = messages[-1]
        channel = primary_msg.channel
        channel_name = f"#{channel.name}" if hasattr(channel, "name") else str(channel_id)
        logger.info(f"Debounce finished. Merged {len(messages)} messages in channel {channel_name} for processing.")

        full_content_lines = []
        all_attachments = []

        for msg in messages:
            user_prefix = f"{msg.author.display_name}: "
            full_content_lines.append(user_prefix + msg.clean_content)

            for att in msg.attachments:
                att_data = {"filename": att.filename, "content": None}
                if att.content_type and (att.content_type.startswith("text/") or att.filename.endswith((".py", ".json", ".toml", ".yaml", ".ini", ".md", ".js", ".ts"))):
                    try:
                        bytes_data = await att.read()
                        att_data["content"] = bytes_data.decode("utf-8", errors="ignore")
                    except Exception as e:
                        logger.warning(f"Failed to read text attachment {att.filename}: {e}")
                all_attachments.append(att_data)

        ingest_content = "\n".join(full_content_lines)
        timestamp_local = primary_msg.created_at.astimezone(self.timezone)
        
        # QMDインジェスト
        await self.context.qmd_engine.ingest_message(
            message_id=str(primary_msg.id),
            timestamp=timestamp_local,
            channel_id=str(channel.id),
            channel_name=channel.name if not isinstance(channel, discord.Thread) else channel.parent.name,
            user_id=str(primary_msg.author.id),
            username=primary_msg.author.display_name,
            content=ingest_content,
            attachments=all_attachments
        )

        has_mention = self.bot.user.mentioned_in(primary_msg) or (
            primary_msg.reference and 
            primary_msg.reference.cached_message and 
            primary_msg.reference.cached_message.author == self.bot.user
        )

        has_attachments = len(all_attachments) > 0

        matched_keyword = None
        for kw in config.trigger_keywords:
            if kw in ingest_content:
                matched_keyword = kw
                break
        has_trigger_keyword = matched_keyword is not None
        is_triggered = has_mention or has_attachments or has_trigger_keyword

        if not is_triggered:
            logger.debug(f"Message {primary_msg.id} skipped: No trigger detected.")
            return

        reasons = []
        if has_mention: reasons.append("メンション検知")
        if has_trigger_keyword: reasons.append(f"キーワード '{matched_keyword}' を検出")
        if has_attachments: reasons.append("添付ファイル検知")
        
        reason_str = " / ".join(reasons)
        logger.info(f"Message {primary_msg.id} triggered LLM evaluation. (Reason: {reason_str})")

        async def run_llm_chain():
            status_msg = None
            try:
                status_embed = create_status_embed("Thinking... 🤔", "メッセージを分析しています...")
                current_task = asyncio.current_task()
                status_msg = await channel.send(embed=status_embed)
                view = CancelView(current_task, status_msg)
                await status_msg.edit(view=view)

                context_data = await self.context.memory_retriever.get_context(primary_msg.clean_content, [
                    {"message_id": str(m.id), "username": m.author.display_name, "content": m.clean_content} for m in messages
                ])

                # データベースから、今回のメッセージを除く直近15件のメッセージを取得
                recent_ids = [str(m.id) for m in messages]
                placeholders = ",".join(["?"] * len(recent_ids))
                
                cursor = await self.context.db_conn.execute(f"""
                SELECT username, content, timestamp
                FROM chat_history
                WHERE channel_id = ? AND message_id NOT IN ({placeholders})
                ORDER BY timestamp DESC
                LIMIT 15
                """, (str(channel.id), *recent_ids))
                history_rows = await cursor.fetchall()
                
                # 時系列順（昇順）に並べ替え
                history_rows = sorted(history_rows, key=lambda r: r["timestamp"])
                
                recent_history_lines = []
                for row in history_rows:
                    recent_history_lines.append(f"{row['username']}: {row['content']}")
                
                # デバウンスバッファに複数メッセージがある場合、最後の1つ以外を末尾に追加
                for msg in messages[:-1]:
                    recent_history_lines.append(f"{msg.author.display_name}: {msg.clean_content}")
                    
                recent_history = "\n".join(recent_history_lines)

                image_parts = []
                for msg in messages:
                    for att in msg.attachments:
                        if att.content_type and att.content_type.startswith("image/"):
                            try:
                                img_bytes = await att.read()
                                part = types.Part.from_bytes(data=img_bytes, mime_type=att.content_type)
                                image_parts.append(part)
                            except Exception as e:
                                logger.warning(f"Failed to load image {att.filename}: {e}")

                current_message_with_attachments = primary_msg.clean_content
                for att in all_attachments:
                    if att["content"] is not None:
                        current_message_with_attachments += f"\n\n--- 添付ファイル: {att['filename']} ---\n{att['content']}"

                now_iso = datetime.datetime.now(self.timezone).isoformat()
                decision = None
                triggered_fallback = False

                try:
                    decision = await self.context.agent.evaluate_and_reply(
                        context_data,
                        recent_history,
                        current_message_with_attachments,
                        channel.name,
                        str(primary_msg.id),
                        now_iso,
                        image_parts
                    )
                except Exception as e:
                    logger.warning(f"Fallback triggered by evaluate_and_reply execution/parse error: {e}")
                    triggered_fallback = True

                if decision is not None:
                    if decision.new_schedule:
                        ns = decision.new_schedule
                        try:
                            task_id = await self.context.scheduler.add_task(
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
                            logger.error(f"Failed to auto-schedule task: {e}")
                            await channel.send("⚠️ スケジュールタスクの自動登録に失敗しました。")

                    if not decision.should_respond:
                        try:
                            await status_msg.delete()
                        except discord.errors.NotFound:
                            pass
                        return

                    if decision.requires_escalation or decision.confidence_score <= 3:
                        triggered_fallback = True
                        logger.info(f"Fallback triggered by self-evaluation: confidence_score={decision.confidence_score}, requires_escalation={decision.requires_escalation}")

                status_embed.title = "Generating reply... ✍️"
                status_embed.description = "返答を作成しています..."
                try:
                    await status_msg.edit(embed=status_embed)
                except discord.errors.NotFound:
                    pass

                reply = None
                if triggered_fallback:
                    logger.info(f"Escalating query response generation to premium model ({config.model_premium})...")
                    status_embed.title = "Re-evaluating Response... 🚀"
                    status_embed.description = "回答を再評価し、高性能モデルで再生成しています..."
                    status_embed.color = 0xe67e22
                    try:
                        await status_msg.edit(embed=status_embed)
                    except discord.errors.NotFound:
                        pass
                    
                    reply = await self.context.agent.generate_reply(
                        context_data, 
                        recent_history, 
                        current_message_with_attachments,
                        channel.name,
                        str(primary_msg.id), 
                        config.model_premium, 
                        image_parts
                    )
                else:
                    reply = AgentReply(
                        reply_content=decision.reply_content,
                        attachment_content=decision.attachment_content,
                        attachment_filename=decision.attachment_filename
                    )

                file_to_send = None
                if reply.attachment_content and reply.attachment_filename:
                    content_bytes = reply.attachment_content
                    if isinstance(content_bytes, str):
                        content_bytes = content_bytes.encode("utf-8")
                    file_io = io.BytesIO(content_bytes)
                    file_to_send = discord.File(file_io, filename=reply.attachment_filename)
                
                main_content = reply.reply_content
                if len(main_content) > 2000:
                    file_io = io.BytesIO(main_content.encode("utf-8"))
                    file_to_send = discord.File(file_io, filename="reply_long.txt")
                    main_content = "⚠️ 返答が長文（2,000文字超）のため、テキストファイルに変換して添付しました。"

                if file_to_send:
                    await channel.send(content=main_content, file=file_to_send)
                else:
                    await channel.send(content=main_content)

                try:
                    await status_msg.delete()
                except discord.errors.NotFound:
                    pass

            except asyncio.CancelledError:
                logger.info(f"Message processing task for channel {channel_id} was cancelled.")
                if status_msg:
                    try:
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
                        await asyncio.sleep(30)
                        await status_msg.delete()
                    except discord.errors.NotFound:
                        pass

        asyncio.create_task(run_llm_chain())

async def setup(bot, context):
    """Cogをロードするための関数"""
    await bot.add_cog(AgentCog(bot, context))
