import datetime
import json
import pytz
import aiosqlite
import logging
from croniter import croniter

from pydantic import BaseModel, Field
from typing import Optional

from src.config import config

class ScheduledTask(BaseModel):
    id: int
    cron_expression: Optional[str] = None
    run_at: Optional[str] = None
    instruction: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[str] = None
    channel_id: str
    next_run: str
    user_id: Optional[str] = None

logger = logging.getLogger("scheduler")

class SchedulerToolRegistry:
    """定期タスクとして実行可能なプログラム関数(Tool)を管理するレジストリ"""
    def __init__(self):
        self._registry = {}

    def register(self, name: str):
        """ツールをレジストリに登録するデコレータ"""
        def decorator(func):
            self._registry[name] = func
            return func
        return decorator

    async def execute(self, name: str, args_json: str, bot, channel_id: str):
        """登録されたツール関数を実行する"""
        if name not in self._registry:
            raise ValueError(f"Tool '{name}' is not registered in SchedulerToolRegistry.")
        
        args = {}
        if args_json:
            try:
                args = json.loads(args_json)
            except Exception as e:
                logger.warning(f"Failed to parse tool args JSON: {e}")
                
        func = self._registry[name]
        # ツール関数を非同期実行 (シグニチャ: bot, channel_id, **args)
        await func(bot, channel_id, **args)

# グローバルなツールレジストリのインスタンス
tool_registry = SchedulerToolRegistry()

# 定期実行ツールテスト用の関数登録
@tool_registry.register("example_ping")
async def example_ping_tool(bot, channel_id: str, **kwargs):
    channel = bot.get_channel(int(channel_id))
    if channel:
        await channel.send("Pong! 定期実行ツールテストが動作しました。")

class TaskScheduler:
    """タイムゾーン対応のタスクスケジューラー (SQLite & croniter 統合)"""
    def __init__(self, db_conn):
        self.config = config
        self.timezone = pytz.timezone(config.timezone)
        self.conn = db_conn

    def calculate_next_run(self, cron_expression: str = None, run_at_iso: str = None) -> datetime.datetime:
        """cron式またはISO日時文字列に基づいて、設定タイムゾーンに沿った次回実行予定日時を算出する"""
        now = datetime.datetime.now(self.timezone)
        
        if cron_expression:
            # croniterはタイムゾーンを剥いで計算し、再度適用
            now_naive = now.replace(tzinfo=None)
            iter = croniter(cron_expression, now_naive)
            next_run_naive = iter.get_next(datetime.datetime)
            return self.timezone.localize(next_run_naive)
        elif run_at_iso:
            dt = datetime.datetime.fromisoformat(run_at_iso)
            if dt.tzinfo is None:
                dt = self.timezone.localize(dt)
            else:
                dt = dt.astimezone(self.timezone)

            # --- 過去日付に対する自動補正セーフガード ---
            if dt < now:
                # LLMが過去の日付（例: 2023年など）を返した場合、時分秒を維持したまま現在日付、あるいは明日へ補正する
                try:
                    candidate_dt = datetime.datetime(
                        year=now.year,
                        month=now.month,
                        day=now.day,
                        hour=dt.hour,
                        minute=dt.minute,
                        second=dt.second,
                        microsecond=dt.microsecond,
                        tzinfo=dt.tzinfo
                    )
                    if candidate_dt >= now:
                        logger.info(f"[Scheduler Safeguard] Corrected past run_at {run_at_iso} to current date: {candidate_dt.isoformat()}")
                        dt = candidate_dt
                    else:
                        # 本日の指定時刻がすでに過ぎている場合は翌日のその時刻にする
                        next_day_dt = candidate_dt + datetime.timedelta(days=1)
                        logger.info(f"[Scheduler Safeguard] Corrected past run_at {run_at_iso} to next day: {next_day_dt.isoformat()}")
                        dt = next_day_dt
                except Exception as e:
                    logger.warning(f"Failed to apply scheduler safeguard to {run_at_iso}: {e}")
            # --------------------------------------------
            return dt
        else:
            raise ValueError("Either 'cron_expression' or 'run_at_iso' must be provided.")

    async def add_task(
        self,
        channel_id: str,
        user_id: str = None,
        cron_expression: str = None,
        run_at: str = None,
        instruction: str = None,
        tool_name: str = None,
        tool_args: str = None
    ) -> int:
        """タスクをSQLiteに追加し、次回の予定時刻を計算して設定する。タスクIDを返す"""
        next_run_dt = self.calculate_next_run(cron_expression, run_at)
        next_run_iso = next_run_dt.isoformat()

        cursor = await self.conn.execute("""
        INSERT INTO scheduled_tasks 
        (cron_expression, run_at, instruction, tool_name, tool_args, channel_id, next_run, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cron_expression,
            run_at,
            instruction,
            tool_name,
            tool_args,
            channel_id,
            next_run_iso,
            user_id
        ))
        await self.conn.commit()
        task_id = cursor.lastrowid
            
        return task_id

    async def delete_task(self, task_id: int) -> bool:
        """タスクを削除する。削除に成功した場合は True を返す"""
        cursor = await self.conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await self.conn.commit()
        return cursor.rowcount > 0

    async def list_tasks(self) -> list[ScheduledTask]:
        """登録されている全スケジュールタスクをリストで取得する"""
        cursor = await self.conn.execute("""
        SELECT id, cron_expression, run_at, instruction, tool_name, tool_args, channel_id, next_run, user_id
        FROM scheduled_tasks
        ORDER BY next_run ASC
        """)
        rows = await cursor.fetchall()
            
        return [ScheduledTask(**row) for row in rows]

    async def get_triggerable_tasks(self) -> list[ScheduledTask]:
        """現在時刻が次回予定時刻(next_run)を過ぎているタスクをフェッチする"""
        now_iso = datetime.datetime.now(self.timezone).isoformat()
        cursor = await self.conn.execute("""
        SELECT id, cron_expression, run_at, instruction, tool_name, tool_args, channel_id, next_run, user_id
        FROM scheduled_tasks
        WHERE next_run <= ?
        """, (now_iso,))
        rows = await cursor.fetchall()
            
        return [ScheduledTask(**row) for row in rows]

    async def reschedule_or_delete(self, task_id: int, cron_expression: str):
        """実行完了したタスクを次回時刻に更新する。1回限りのタスクは削除する"""
        if cron_expression:
            next_run_dt = self.calculate_next_run(cron_expression=cron_expression)
            next_run_iso = next_run_dt.isoformat()
            await self.conn.execute("""
            UPDATE scheduled_tasks
            SET next_run = ?
            WHERE id = ?
            """, (next_run_iso, task_id))
            await self.conn.commit()
            logger.info(f"Task ID {task_id} rescheduled to next run: {next_run_iso}.")
        else:
            await self.delete_task(task_id)
