import datetime
import logging
import discord
from discord import app_commands
from discord.ext import commands, tasks
import pytz

from src.config import config
from src.scheduler import tool_registry, ScheduledTask

logger = logging.getLogger("scheduler_cog")

class SchedulerCog(commands.Cog):
    """スケジュール機能（定期タスク・ポーリング）を管理する Cog"""
    def __init__(self, bot, context):
        self.bot = bot
        self.context = context
        self.timezone = pytz.timezone(config.timezone)
        
    async def cog_load(self):
        """Cogがロードされたときにポーリングを開始する"""
        if not self.poll_scheduled_tasks.is_running():
            self.poll_scheduled_tasks.start()
        logger.info("Scheduler polling loop started.")

    async def cog_unload(self):
        """Cogがアンロードされたときにポーリングを停止する"""
        self.poll_scheduled_tasks.cancel()
        logger.info("Scheduler polling loop stopped.")

    @tasks.loop(minutes=1.0)
    async def poll_scheduled_tasks(self):
        logger.debug("Checking for triggerable scheduled tasks...")
        try:
            triggerable_tasks = await self.context.scheduler.get_triggerable_tasks()
            for t in triggerable_tasks:
                task_id = t.id
                cron_expr = t.cron_expression
                channel_id = t.channel_id
                instruction = t.instruction
                tool_name = t.tool_name
                tool_args = t.tool_args

                channel = self.bot.get_channel(int(channel_id))
                if not channel:
                    logger.warning(f"Channel {channel_id} not found for task {task_id}.")
                    await self.context.scheduler.reschedule_or_delete(task_id, cron_expr)
                    continue

                if tool_name:
                    logger.info(f"Triggering scheduled task ID {task_id} in #{channel.name}. (Tool: '{tool_name}')")
                    try:
                        await tool_registry.execute(tool_name, tool_args, self.bot, channel_id)
                    except Exception as e:
                        logger.error(f"Failed to execute tool {tool_name}: {e}")
                        await channel.send(f"⚠️ 定期タスク（ツール: {tool_name}）の実行中にエラーが発生しました。")
                
                elif instruction:
                    logger.info(f"Triggering scheduled task ID {task_id} in #{channel.name}. (Instruction: '{instruction}')")
                    task_user_id = t.user_id
                    
                    context_data = await self.context.memory_retriever.get_context(instruction, [])
                    try:
                        reply = await self.context.agent.generate_scheduled_reply(context_data, instruction)
                        
                        if task_user_id:
                            import re
                            mention_str = f"<@{task_user_id}>"
                            reply_stripped = reply.strip()
                            if reply_stripped.startswith("@"):
                                reply = re.sub(r'^@\w+', mention_str, reply_stripped, count=1)
                            else:
                                reply = f"{mention_str}\n\n{reply}"
                                
                        await channel.send(reply)
                    except Exception as e:
                        logger.error(f"Failed to generate scheduled reply: {e}")
                        await channel.send("⚠️ 定期タスクのメッセージ生成中にエラーが発生しました。")

                await self.context.scheduler.reschedule_or_delete(task_id, cron_expr)

        except Exception as e:
            logger.error(f"Exception in scheduler loop: {e}")

    @app_commands.command(name="schedule_list", description="登録されている定期タスクの一覧を表示します。")
    async def schedule_list(self, interaction: discord.Interaction):
        logger.info(f"User {interaction.user.display_name} executed slash command /schedule_list")
        await interaction.response.defer()
        try:
            tasks_list = await self.context.scheduler.list_tasks()
            if not tasks_list:
                await interaction.followup.send("現在登録されているスケジュールタスクはありません。")
                return

            embed = discord.Embed(
                title="📅 スケジュールタスク一覧",
                color=0x3498db,
                timestamp=datetime.datetime.now(self.timezone)
            )
            for t in tasks_list:
                schedule_str = f"cron: `{t.cron_expression}`" if t.cron_expression else f"日時: `{t.run_at}`"
                action_str = f"指示: {t.instruction}" if t.instruction else f"ツール: {t.tool_name}"
                embed.add_field(
                    name=f"ID: {t.id} | チャンネル: <#{t.channel_id}>",
                    value=f"{schedule_str}\n{action_str}\n次回実行: `{t.next_run}`",
                    inline=False
                )
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"❌ タスク一覧の取得中にエラーが発生しました: {e}")

    @app_commands.command(name="schedule_delete", description="指定したIDの定期タスクを削除します。")
    @app_commands.describe(task_id="削除するタスクのID")
    async def schedule_delete(self, interaction: discord.Interaction, task_id: int):
        logger.info(f"User {interaction.user.display_name} executed slash command /schedule_delete (Parameters: task_id={task_id})")
        await interaction.response.defer()
        try:
            success = await self.context.scheduler.delete_task(task_id)
            if success:
                await interaction.followup.send(f"✅ 定期タスク (ID: {task_id}) を削除しました。")
            else:
                await interaction.followup.send(f"❌ 指定されたID (ID: {task_id}) のタスクが見つかりませんでした。")
        except Exception as e:
            await interaction.followup.send(f"❌ タスクの削除中にエラーが発生しました: {e}")

    @app_commands.command(name="schedule_add", description="定期タスクを手動で登録します。")
    @app_commands.describe(
        cron="cron式 (例: '0 9 * * 1' 毎週月曜9時)",
        instruction="実行させる自然言語指示 (例: '今日の天気を要約して')"
    )
    async def schedule_add(self, interaction: discord.Interaction, cron: str, instruction: str):
        logger.info(f"User {interaction.user.display_name} executed slash command /schedule_add (Parameters: cron={cron}, instruction={instruction})")
        await interaction.response.defer()
        try:
            self.context.scheduler.calculate_next_run(cron_expression=cron)
            task_id = await self.context.scheduler.add_task(
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

async def setup(bot, context):
    """メインから呼ばれるセットアップ関数"""
    await bot.add_cog(SchedulerCog(bot, context))
