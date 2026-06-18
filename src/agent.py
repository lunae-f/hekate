import json
import logging
import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from src.config import config

logger = logging.getLogger("agent")

class TaskScheduleInstruction(BaseModel):
    cron_expression: str = Field(description="cron指定。例: '0 9 * * 1' (毎週月曜9時)。一回限りの場合は空文字にする")
    run_at: Optional[str] = Field(
        None, 
        description=(
            "一回限りのタスクの場合の実行予定日時 (ISO8601形式)。不要なら省略。"
            "必ずプロンプトに指定された『現在日時』の年月日を基準とし、過去の日時（例: 2023年など）を設定しないこと。"
        )
    )
    instruction: Optional[str] = Field(None, description="定期実行時にLLMに実行させる自然言語指示。ユーザーへ通知・送信するメッセージ内容など")
    tool_name: Optional[str] = Field(None, description="プログラム実行用のツール名 (将来用)。不要なら省略")
    tool_args: Optional[str] = Field(None, description="ツール実行用のJSONパラメータ (将来用)。不要なら省略")

class AgentEvaluationAndReply(BaseModel):
    internal_monologue: str = Field(description="現在の会話状況の分析と応答要否・難易度評価の思考プロセス")
    should_respond: bool = Field(description="自身が応答すべき、またはタスク予約を実行すべきと判断した場合は True")
    requires_escalation: bool = Field(description="質問が高度・難解で、高性能モデル（gemini-3.5-flash）による再生成が必要だと判断した場合は True。これが True の場合、reply_content は必ず空文字（\"\"）にすること")
    confidence_score: int = Field(description="この回答に対する自身の確信度（1〜5の整数。5が最高。requires_escalation が True の場合は 1 にすること）")
    reply_content: str = Field("", description="返答のメインメッセージ（requires_escalation が True の場合は必ず空文字にします）")
    attachment_content: Optional[str] = Field(None, description="添付テキストファイルの内容。不要なら省略（requires_escalation が True の場合は省略）")
    attachment_filename: Optional[str] = Field(None, description="添付ファイルのファイル名。不要なら省略")
    new_schedule: Optional[TaskScheduleInstruction] = Field(None, description="スケジュール情報。不要なら省略")

class AgentReply(BaseModel):
    reply_content: str = Field(description="返答のメインメッセージ。ファイルを添付する場合はその説明")
    attachment_content: Optional[str] = Field(None, description="添付テキストファイルとして送信したい長文内容 (ソースコードやログ等)。不要なら省略")
    attachment_filename: Optional[Optional[str]] = Field(None, description="添付ファイルのファイル名 (例: 'code.py', 'result.txt')。添付がある場合は必須")

class AIAgent:
    """Gemini API を用いた思考判定と返答生成エージェント"""
    def __init__(self):
        self.config = config
        self.client = genai.Client()

    async def evaluate_and_reply(
        self,
        context: str,
        recent_history: str,
        current_message: str,
        message_id: str,
        current_time_iso: str,
        image_parts: list = None
    ) -> AgentEvaluationAndReply:
        """
        gemini-3.1-flash-lite (安価モデル) を用いて、応答要否、エスカレーション判定、および一次返答生成を1回で同時に行う
        """
        logger.info(f"Evaluating and generating response for message {message_id} with {self.config.evaluator_model}...")
        logger.debug(f"[Evaluator] current_time_iso={current_time_iso}")

        # 現在日時のパースと曜日取得
        try:
            dt = datetime.datetime.fromisoformat(current_time_iso)
            wdays = ["月", "火", "水", "木", "金", "土", "日"]
            wday_str = wdays[dt.weekday()]
            time_display = f"{current_time_iso} ({dt.year}年{dt.month:02d}月{dt.day:02d}日 {wday_str}曜日)"
            example_iso = f"{dt.year}-{dt.month:02d}-{dt.day:02d}T12:30:00+09:00"
        except Exception as e:
            logger.warning(f"Failed to format current_time_iso: {e}")
            time_display = current_time_iso
            example_iso = "2026-06-18T12:30:00+09:00"

        prompt = f"""
current_time: {time_display}

以下は過去の文脈および直近の会話履歴です。これらを踏まえて、最新メッセージに対して自分が応答すべきか、応答する場合に返答メッセージを生成し、かつより高性能なモデル（gemini-3.5-flash）へのエスカレーションが必要であるかを判定してください。

■ エスカレーション (requires_escalation) の判定基準:
- 質問がプログラミング、コードデバッグ、エラー解説、複雑なシステム設計や論理的推論を必要とする場合、必ず `requires_escalation: true` としてください。
- 挨拶、簡単な雑談、単純な質問、一言の返答で済む軽い対話の場合は `requires_escalation: false` としてください。
- 【重要】`requires_escalation: true` の場合、無駄なトークン消費を防ぐため、`reply_content` は必ず空文字（""）にしてください。

もし定期タスクの登録依頼である場合は、必ず上記「現在日時」を基準にしてスケジュール（cron_expression または run_at）を正確に解釈・抽出してください。
相対的な日時（例：「12:30になったら」）は、現在日時の日付部分を継承し、将来の正確な絶対日時（例：{example_iso} などのISO8601形式）として登録する必要があります。

# 過去の関連文脈 (想起記憶)
{context}

# 直近の会話履歴 (短期記憶)
{recent_history}

# 最新メッセージ
{current_message}
"""
        logger.debug(f"[Evaluator Prompt]\n{prompt}")
        contents = []
        if image_parts:
            contents.extend(image_parts)
        contents.append(prompt)

        response = await self.client.aio.models.generate_content(
            model=self.config.evaluator_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.config.evaluator_instruction,
                temperature=self.config.temperature,
                response_mime_type="application/json",
                response_schema=AgentEvaluationAndReply,
            )
        )
        
        logger.debug(f"[Evaluator Raw Response] {response.text}")
        
        # 構造化JSONをパースしてPydanticオブジェクトに復元
        data = json.loads(response.text)
        result = AgentEvaluationAndReply(**data)
        logger.info(f"Evaluation: should_respond={result.should_respond}, requires_escalation={result.requires_escalation}, confidence_score={result.confidence_score}. Monologue: '{result.internal_monologue}'")
        return result

    async def generate_reply(
        self,
        context: str,
        recent_history: str,
        current_message: str,
        message_id: str,
        model_name: str,
        image_parts: list = None
    ) -> AgentReply:
        """
        指定されたモデルを用いて、キャラクター設定に沿った返答と必要に応じたファイル添付データを生成する
        """
        logger.info(f"Generating reply with {model_name}...")

        prompt = f"""
以下は過去の関連文脈および直近の会話履歴です。最新のメッセージに対して、キャラクター設定に従って適切な返答を生成してください。長文コードなどを出力する場合は、適宜アタッチメントファイル（attachment_content）に格納して出力してください。

# 過去の関連文脈 (想起記憶)
{context}

# 直近の会話履歴 (短期記憶)
{recent_history}

# 最新メッセージ
{current_message}
"""
        contents = []
        if image_parts:
            contents.extend(image_parts)
        contents.append(prompt)

        response = await self.client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.config.generator_instruction,
                temperature=self.config.temperature,
                response_mime_type="application/json",
                response_schema=AgentReply,
            )
        )
        
        data = json.loads(response.text)
        reply = AgentReply(**data)

        # 添付ファイル情報
        att_info = ""
        if reply.attachment_filename:
            size_bytes = len(reply.attachment_content.encode("utf-8")) if reply.attachment_content else 0
            att_info = f", Attachment: '{reply.attachment_filename}' [{size_bytes} bytes]"
        
        reply_len = len(reply.reply_content) if reply.reply_content else 0
        logger.info(f"Reply generated successfully. (Length: {reply_len} chars{att_info})")

        return reply

    async def generate_scheduled_reply(self, context: str, instruction: str) -> str:
        """
        定期タスク実行時に、指定された指示（instruction）と文脈からDiscord投稿用メッセージを生成する
        """
        prompt = f"""
あなたは定期実行タスクを担当するエージェントです。以下の指示（Instruction）および想起された関連文脈に基づいて、Discordに投稿するためのメッセージを生成してください。

# 指示（Instruction）
{instruction}

# 過去の関連文脈
{context}
"""
        response = await self.client.aio.models.generate_content(
            model=self.config.generator_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.config.generator_instruction,
                temperature=self.config.temperature,
            )
        )
        return response.text
