import datetime
import json
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
from src.agent import AgentReply, GeneratedResponse
from src.ui import CancelView, create_status_embed

logger = logging.getLogger("agent_cog")

class AgentCog(commands.Cog):
    """エージェントメッセージ処理、デバウンス、想起、一次判定・応答生成、長期記憶要約バッチを管理する Cog"""
    def __init__(self, bot, context):
        self.bot = bot
        self.context = context
        self.timezone = pytz.timezone(config.timezone)
        self.debounce_buffers = {}
        self.ignore_all = False
        self.ignored_channels = set()

    async def cog_load(self):
        """Cogロード時に深夜メモリ要約を開始し、設定をロードする"""
        if not self.daily_memory_consolidation.is_running():
            self.daily_memory_consolidation.start()
        logger.info("Daily memory consolidation task started.")

        try:
            cursor = await self.context.db_conn.execute("SELECT value FROM settings WHERE key = 'ignore_all'")
            row = await cursor.fetchone()
            self.ignore_all = (row["value"] == "true") if row else False

            cursor = await self.context.db_conn.execute("SELECT channel_id FROM ignored_channels")
            rows = await cursor.fetchall()
            self.ignored_channels = {r["channel_id"] for r in rows}
            logger.info(f"Loaded ignore settings: ignore_all={self.ignore_all}, ignored_channels={self.ignored_channels}")
        except Exception as e:
            logger.error(f"Failed to load ignore settings: {e}")

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

    # ignoreコマンドグループの定義
    ignore_group = app_commands.Group(name="ignore", description="メッセージの監視無視（応答除外）設定を管理します。")

    @ignore_group.command(name="channel", description="指定したチャンネルの自動応答を無視/無視解除します。")
    @app_commands.describe(
        action="無視リストへの操作を選択します。",
        target_channel="対象のチャンネル（指定しない場合は現在のチャンネル）"
    )
    @app_commands.choices(action=[
        app_commands.Choice(name="無視リストに追加", value="add"),
        app_commands.Choice(name="無視リストから削除", value="remove")
    ])
    async def ignore_channel(
        self,
        interaction: discord.Interaction,
        action: str,
        target_channel: discord.abc.GuildChannel = None
    ):
        logger.info(f"User {interaction.user.display_name} executed slash command /ignore channel action={action}")
        await interaction.response.defer(ephemeral=True)
        
        # チャンネルの決定
        chan = target_channel or interaction.channel
        if not chan:
            await interaction.followup.send("❌ チャンネルが特定できませんでした。")
            return
            
        chan_id = str(chan.id)
        chan_name = chan.name

        try:
            if action == "add":
                await self.context.db_conn.execute(
                    "INSERT OR REPLACE INTO ignored_channels (channel_id, channel_name) VALUES (?, ?)",
                    (chan_id, chan_name)
                )
                await self.context.db_conn.commit()
                self.ignored_channels.add(chan_id)
                await interaction.followup.send(f"✅ チャンネル <#{chan_id}> を無視リストに追加しました。以降、メンション時を除き、自動応答（キーワード等）を無視します。")
            else:
                await self.context.db_conn.execute(
                    "DELETE FROM ignored_channels WHERE channel_id = ?",
                    (chan_id,)
                )
                await self.context.db_conn.commit()
                self.ignored_channels.discard(chan_id)
                await interaction.followup.send(f"✅ チャンネル <#{chan_id}> を無視リストから削除しました。通常通り自動応答します。")
        except Exception as e:
            logger.error(f"Failed to update channel ignore settings: {e}")
            await interaction.followup.send(f"❌ 設定の更新に失敗しました: {e}")

    @ignore_group.command(name="global", description="全チャンネルにおける自動応答の一括無視を切り替えます。")
    @app_commands.describe(enabled="Trueで全チャンネル無視、Falseで解除")
    async def ignore_global(self, interaction: discord.Interaction, enabled: bool):
        logger.info(f"User {interaction.user.display_name} executed slash command /ignore global enabled={enabled}")
        await interaction.response.defer(ephemeral=True)

        val_str = "true" if enabled else "false"
        try:
            await self.context.db_conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('ignore_all', ?)",
                (val_str,)
            )
            await self.context.db_conn.commit()
            self.ignore_all = enabled
            
            status_text = "有効（一括無視）" if enabled else "無効（通常動作）"
            await interaction.followup.send(f"✅ グローバル無視設定を **{status_text}** に変更しました。")
        except Exception as e:
            logger.error(f"Failed to update global ignore settings: {e}")
            await interaction.followup.send(f"❌ 設定の更新に失敗しました: {e}")

    @ignore_group.command(name="status", description="現在のメッセージ無視設定の一覧を確認します。")
    async def ignore_status(self, interaction: discord.Interaction):
        logger.info(f"User {interaction.user.display_name} executed slash command /ignore status")
        await interaction.response.defer(ephemeral=True)

        try:
            cursor = await self.context.db_conn.execute("SELECT channel_id, channel_name FROM ignored_channels")
            rows = await cursor.fetchall()
            
            embed = discord.Embed(title="⚙️ メッセージ無視（応答除外）設定ステータス", color=0x3498db)
            
            g_status = "🔴 有効（全チャンネルでメンション以外無視）" if self.ignore_all else "🟢 無効（通常応答）"
            embed.add_field(name="グローバル無視設定", value=g_status, inline=False)
            
            if rows:
                lines = [f"- <#{r['channel_id']}> (ID: {r['channel_id']})" for r in rows]
                ch_list_str = "\n".join(lines)
            else:
                ch_list_str = "*無視設定されているチャンネルはありません。*"
                
            embed.add_field(name="無視設定チャンネル一覧", value=ch_list_str, inline=False)
            embed.set_footer(text="※無視設定中のチャンネルでも、直接メンションまたは返信された場合は応答します。")
            
            await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Failed to get ignore status: {e}")
            await interaction.followup.send(f"❌ ステータスの取得に失敗しました: {e}")

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
                att_data = {"filename": att.filename, "content": None, "local_path": None}
                
                # 画像の判定
                is_image = False
                mime_type = att.content_type
                if mime_type and mime_type.startswith("image/"):
                    is_image = True
                else:
                    ext = att.filename.split(".")[-1].lower() if "." in att.filename else ""
                    if ext in ("png", "jpg", "jpeg", "webp", "gif"):
                        is_image = True
                        mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"

                # 画像の場合、ローカルに保存
                if is_image:
                    try:
                        img_bytes = await att.read()
                        save_dir = Path(__file__).parent.parent.parent / "data" / "attachments"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        save_name = f"{msg.id}_{att.filename}"
                        save_path = save_dir / save_name
                        
                        async with aiofiles.open(save_path, mode="wb") as f_img:
                            await f_img.write(img_bytes)
                            
                        att_data["local_path"] = f"data/attachments/{save_name}"
                        logger.info(f"Saved attachment image to {att_data['local_path']}")
                    except Exception as e:
                        logger.warning(f"Failed to save image attachment {att.filename}: {e}")

                # テキスト系ファイルの場合、中身を読み込む
                elif att.content_type and (att.content_type.startswith("text/") or att.filename.endswith((".py", ".json", ".toml", ".yaml", ".ini", ".md", ".js", ".ts"))):
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

        matched_keyword = None
        for kw in config.trigger_keywords:
            if kw in ingest_content:
                matched_keyword = kw
                break
        has_trigger_keyword = matched_keyword is not None
        
        # 添付ファイル単体でのトリガーは廃止
        is_triggered = has_mention or has_trigger_keyword

        if not is_triggered:
            logger.debug(f"Message {primary_msg.id} skipped: No trigger detected.")
            return

        # 無視設定のチェック（直接メンションがない場合のみ適用）
        if not has_mention:
            if self.ignore_all:
                logger.info(f"Message {primary_msg.id} ignored: Global ignore is active.")
                return
            if str(channel.id) in self.ignored_channels:
                logger.info(f"Message {primary_msg.id} ignored: Channel {channel_name} is in ignore list.")
                return

            # 自動応答に対する経過件数制限の適用
            elapsed = await self.get_messages_elapsed_since_agent(str(channel.id))
            if elapsed >= config.max_messages_after_agent:
                logger.info(
                    f"Message {primary_msg.id} ignored: "
                    f"Elapsed messages since last agent reply ({elapsed}) "
                    f"exceeds limit ({config.max_messages_after_agent})."
                )
                return

        reasons = []
        if has_mention: reasons.append("メンション検知")
        if has_trigger_keyword: reasons.append(f"キーワード '{matched_keyword}' を検出")
        
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

                # インナー関数: 過去の履歴をデータベースから取得し、画像を含む時系列リストとしてパースする
                async def load_recent_history_parts(offset: int, date_str: str = None) -> list:
                    recent_ids = [str(m.id) for m in messages]
                    placeholders = ",".join(["?"] * len(recent_ids))
                    
                    history_rows = []
                    try:
                        if date_str:
                            # 特定の日付 (YYYY-MM-DD) のログを最大30件取得
                            date_start = f"{date_str}T00:00:00"
                            date_end = f"{date_str}T23:59:59"
                            cursor_hist = await self.context.db_conn.execute(f"""
                            SELECT message_id, username, content, timestamp, attachments
                            FROM chat_history
                            WHERE channel_id = ? AND timestamp >= ? AND timestamp <= ? AND message_id NOT IN ({placeholders})
                            ORDER BY timestamp DESC
                            LIMIT 30
                            """, (str(channel.id), date_start, date_end, *recent_ids))
                            history_rows = await cursor_hist.fetchall()
                        else:
                            # 件数ベースでの過去遡り取得
                            cursor_hist = await self.context.db_conn.execute(f"""
                            SELECT message_id, username, content, timestamp, attachments
                            FROM chat_history
                            WHERE channel_id = ? AND message_id NOT IN ({placeholders})
                            ORDER BY timestamp DESC
                            LIMIT ?
                            """, (str(channel.id), *recent_ids, offset))
                            history_rows = await cursor_hist.fetchall()
                    except Exception as e_db:
                        logger.error(f"Failed to query chat_history for context: {e_db}")
                        
                    # 時系列順（昇順）に並べ替え
                    history_rows = sorted(history_rows, key=lambda r: r["timestamp"])
                    
                    parts = []
                    for row in history_rows:
                        # メッセージテキストを追加
                        parts.append(f"{row['username']}: {row['content']}")
                        
                        # 画像アタッチメントがあればローカルからロード
                        if row["attachments"]:
                            try:
                                atts = json.loads(row["attachments"])
                                for att in atts:
                                    if isinstance(att, dict):
                                        local_path = att.get("local_path")
                                        if local_path:
                                            p_path = Path(__file__).parent.parent.parent / local_path
                                            if p_path.exists():
                                                async with aiofiles.open(p_path, mode="rb") as f_img:
                                                    img_bytes = await f_img.read()
                                                ext = local_path.split(".")[-1].lower()
                                                mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
                                                part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
                                                parts.append(part)
                                                logger.info(f"Loaded past image context: {local_path}")
                                    elif isinstance(att, str):
                                        logger.info(f"Legacy attachment format (string): {att}. Skipping loading file contents.")
                            except Exception as e_load:
                                logger.warning(f"Failed to load past image for message {row['message_id']}: {e_load}")
                                
                    # 今回のデバウンスしたメッセージバッファ（最後の1個以外）と画像も追加
                    for msg in messages[:-1]:
                        parts.append(f"{msg.author.display_name}: {msg.clean_content}")
                        for att in msg.attachments:
                            ext = att.filename.split(".")[-1].lower() if "." in att.filename else ""
                            if (att.content_type and att.content_type.startswith("image/")) or ext in ("png", "jpg", "jpeg", "webp", "gif"):
                                try:
                                    img_bytes = await att.read()
                                    mime = att.content_type or f"image/{'jpeg' if ext == 'jpg' else ext}"
                                    part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
                                    parts.append(part)
                                except Exception as e_img:
                                    logger.warning(f"Failed to load debounce message image: {e_img}")
                                    
                    return parts

                # 初期ロード（直近15件）
                current_offset = 15
                recent_history_parts = await load_recent_history_parts(current_offset)

                image_parts = []
                for att in primary_msg.attachments:
                    ext = att.filename.split(".")[-1].lower() if "." in att.filename else ""
                    if (att.content_type and att.content_type.startswith("image/")) or ext in ("png", "jpg", "jpeg", "webp", "gif"):
                        try:
                            img_bytes = await att.read()
                            mime = att.content_type or f"image/{'jpeg' if ext == 'jpg' else ext}"
                            part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
                            image_parts.append(part)
                        except Exception as e_img:
                            logger.warning(f"Failed to load primary message image: {e_img}")

                current_message_with_attachments = primary_msg.clean_content
                for att in all_attachments:
                    if att["content"] is not None:
                        current_message_with_attachments += f"\n\n--- 添付ファイル: {att['filename']} ---\n{att['content']}"

                now_iso = datetime.datetime.now(self.timezone).isoformat()
                decision = None
                triggered_fallback = False
                
                # マルチターン過去ログ検索ループ
                max_retries = 3
                retry_count = 0
                target_date = None
                
                while retry_count < max_retries:
                    # 2回目以降のループ時に進捗をDiscord上のThinkingにフィードバック
                    if retry_count > 0 and status_msg:
                        try:
                            step_desc = "過去のログを探索しています..."
                            if target_date:
                                step_desc = f"{target_date} のログを追加ロード中..."
                            elif current_offset > 15:
                                step_desc = f"さらに過去のメッセージを遡っています..."
                                
                            status_embed.description = step_desc
                            await status_msg.edit(embed=status_embed)
                        except discord.errors.NotFound:
                            pass
                        except Exception as e_status:
                            logger.warning(f"Failed to update status message: {e_status}")

                    try:
                        decision = await self.context.agent.evaluate_and_reply(
                            context_data,
                            recent_history_parts,
                            current_message_with_attachments,
                            channel.name,
                            str(primary_msg.id),
                            now_iso,
                            image_parts
                        )
                    except Exception as e:
                        logger.warning(f"Fallback triggered by evaluate_and_reply execution/parse error: {e}")
                        triggered_fallback = True
                        break

                    # 追加要求がある場合
                    if decision and decision.requires_more_context and decision.context_request:
                        req = decision.context_request
                        retry_count += 1
                        
                        if req.request_type == "date" and req.target_date:
                            target_date = req.target_date
                            logger.info(f"LLM requested context for date: {target_date} (Reason: {req.reason})")
                            recent_history_parts = await load_recent_history_parts(0, date_str=target_date)
                        else:
                            current_offset += req.offset_count or 15
                            if current_offset > 50:
                                current_offset = 50
                            logger.info(f"LLM requested context offset: +{req.offset_count or 15} (Total offset: {current_offset}) (Reason: {req.reason})")
                            recent_history_parts = await load_recent_history_parts(current_offset)
                    else:
                        break

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

                    if decision.requires_escalation or decision.confidence_score <= 3 or decision.requires_more_context:
                        triggered_fallback = True
                        logger.info(f"Fallback triggered: confidence_score={decision.confidence_score}, requires_escalation={decision.requires_escalation}, requires_more_context={decision.requires_more_context}")

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
                        recent_history_parts, 
                        current_message_with_attachments,
                        channel.name,
                        str(primary_msg.id), 
                        config.model_premium, 
                        image_parts
                    )
                else:
                    reply = GeneratedResponse(
                        reply_content=decision.reply_content,
                        attachment_content=decision.attachment_content,
                        attachment_filename=decision.attachment_filename,
                        sources=None
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

                # 参照ソースがある場合は Embed を追加して送信
                sources_embed = None
                if reply.sources:
                    sources_embed = discord.Embed(
                        title="🔍 参照された情報ソース",
                        color=0x3498db
                    )
                    description_lines = []
                    for idx, src in enumerate(reply.sources, 1):
                        title = src.get("title", "Web Source")
                        uri = src.get("uri", "")
                        if uri:
                            description_lines.append(f"{idx}. [{title}]({uri})")
                        else:
                            description_lines.append(f"{idx}. {title}")
                    sources_embed.description = "\n".join(description_lines)

                if file_to_send:
                    if sources_embed:
                        await channel.send(content=main_content, file=file_to_send, embed=sources_embed)
                    else:
                        await channel.send(content=main_content, file=file_to_send)
                else:
                    if sources_embed:
                        await channel.send(content=main_content, embed=sources_embed)
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

    async def get_messages_elapsed_since_agent(self, channel_id: str) -> int:
        """最後のBot発言から現在までに何件メッセージが経過したかを返す"""
        cursor = await self.context.db_conn.execute("""
            SELECT user_id 
            FROM chat_history 
            WHERE channel_id = ? 
            ORDER BY timestamp DESC, message_id DESC 
            LIMIT 50
        """, (channel_id,))
        rows = await cursor.fetchall()
        
        bot_id_str = str(self.bot.user.id)
        elapsed_count = 0
        found_bot = False
        
        # rows[0] はインジェストしたばかりの最新メッセージなので、rows[1:] から遡る
        for row in rows[1:]:
            if row["user_id"] == bot_id_str:
                found_bot = True
                break
            elapsed_count += 1
            
        if not found_bot:
            # 履歴に一度もBotの発言がない、または遠い過去の場合は上限超えとみなす
            return 9999
            
        return elapsed_count

async def setup(bot, context):
    """Cogをロードするための関数"""
    await bot.add_cog(AgentCog(bot, context))
