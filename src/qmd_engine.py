import os
import json
import asyncio
import datetime
import numpy as np
import aiofiles
from pathlib import Path
from dateutil.relativedelta import relativedelta
import pytz
from google import genai
from sudachipy import dictionary, tokenizer
import logging

from src.config import config
from src.db import get_db_connection

logger = logging.getLogger("qmd")

# 保存ディレクトリ
ROOT_DIR = Path(__file__).parent.parent
MEM_DIR = ROOT_DIR / "data" / "memory"

class SudachiTokenizer:
    """Sudachiを用いた日本語形態素解析トークナイザー (Aモードで最短分割)"""
    def __init__(self):
        self.dict = dictionary.Dictionary()
        self.tokenizer = self.dict.create()
        self.split_mode = tokenizer.Tokenizer.SplitMode.A

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        # 空白や制御文字などをクリーンアップ
        clean_text = " ".join(text.split())
        if not clean_text:
            return []
        
        tokens = self.tokenizer.tokenize(clean_text, self.split_mode)
        # 空白文字を除いた形態素の surface を返す
        return [m.surface() for m in tokens if m.surface().strip()]

async def get_embedding_with_retry(client: genai.Client, text: str, model_name: str) -> list[float]:
    """指数バックオフを用いた Embedding 取得 (Rate Limit対策)"""
    backoff = 1.0
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Client.aio property を使って非同期 API コール
            response = await client.aio.models.embed_content(
                model=model_name,
                contents=text
            )
            if response.embeddings:
                return response.embeddings[0].values
            raise ValueError("No embeddings returned from Gemini API.")
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            # 指数バックオフ (1s, 2s, 4s, 8s, 16s)
            logger.warning(f"Embedding API rate-limited. Retrying in {backoff:.1f}s (Attempt {attempt + 2}/{max_retries})...")
            await asyncio.sleep(backoff)
            backoff *= 2.0

class QMDEngine:
    """QMD (Query Markup Documents) キャッシュおよび 2段階ハイブリッド検索エンジン"""
    def __init__(self, db_conn, gemini_client):
        self.config = config
        self.tokenizer = SudachiTokenizer()
        self.conn = db_conn
        self.client = gemini_client
        self._background_tasks = set()

    async def wait_for_tasks(self):
        """仕掛かり中のバックグラウンドタスクがすべて完了するのを安全に待機する"""
        if self._background_tasks:
            logger.info(f"Waiting for {len(self._background_tasks)} background embedding tasks to complete...")
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            logger.info("All background embedding tasks completed.")

    async def ingest_message(
        self,
        message_id: str,
        timestamp: datetime.datetime,
        channel_id: str,
        channel_name: str,
        user_id: str,
        username: str,
        content: str,
        attachments: list[dict] = None
    ):
        """
        受信メッセージをQMD形式で記録し、ベクトル化を行う。
        1. chat_history にインサート (トークンキャッシュ)
        2. chat_log_YYYY-MM.md (Markdown) に非同期アペンド (バックアップ)
        3. 非同期で Embedding を取得して SQLite に保存 (バースト保護のバックグラウンドタスク)
        """
        attachments_list = attachments or []
        # 添付ファイルのメタデータJSONリスト
        attachments_json = json.dumps(attachments_list, ensure_ascii=False)

        # 1. 日本語トークナイズ
        token_list = self.tokenizer.tokenize(content)
        tokens_str = " ".join(token_list)

        # 2. SQLite への登録
        await self.conn.execute("""
        INSERT OR REPLACE INTO chat_history 
        (message_id, timestamp, channel_id, channel_name, user_id, username, content, tokens, attachments)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message_id,
            timestamp.isoformat(),
            channel_id,
            channel_name,
            user_id,
            username,
            content,
            tokens_str,
            attachments_json
        ))
        await self.conn.commit()

        # 3. Markdownログファイルへのアペンド
        MEM_DIR.mkdir(parents=True, exist_ok=True)
        month_str = timestamp.strftime("%Y-%m")
        md_file = MEM_DIR / f"chat_log_{month_str}.md"

        yaml_meta = [
            "---",
            f"id: {message_id}",
            f"timestamp: {timestamp.isoformat()}",
            f"channel: {channel_name} ({channel_id})",
            f"user: {username} ({user_id})"
        ]
        if attachments_list:
            yaml_meta.append(f"attachments: {attachments_json}")
        yaml_meta.append("---")

        md_content = "\n".join(yaml_meta) + "\n" + content + "\n\n"

        # 添付テキストファイルの中身もログ化
        for att in attachments_list:
            filename = att["filename"]
            file_content = att.get("content", "")
            if file_content:
                _, ext = os.path.splitext(filename)
                lang = ext.replace(".", "") if ext else ""
                md_content += f"[添付ファイル: {filename}]\n```{lang}\n{file_content}\n```\n\n"

        # 非同期でMarkdownに追記
        async with aiofiles.open(md_file, mode="a", encoding="utf-8") as f:
            await f.write(md_content)

        logger.info(f"Message {message_id} ingested to SQLite (tokens cached) and Markdown backup.")

        # 4. 非同期での Embedding 取得タスクを起動 (直列またはバックグラウンド化でAPIバーストを許容)
        task = asyncio.create_task(self._process_embedding(message_id, content))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _process_embedding(self, message_id: str, content: str):
        """バックグラウンドで Embedding を取得して SQLite に保存する"""
        try:
            vector = await get_embedding_with_retry(self.client, content, self.config.embedding_model)
            vector_blob = np.array(vector, dtype=np.float32).tobytes()
            
            await self.conn.execute("""
            INSERT OR REPLACE INTO embeddings (message_id, vector)
            VALUES (?, ?)
            """, (message_id, vector_blob))
            await self.conn.commit()
            logger.info(f"Successfully cached embedding vector for message {message_id}.")
        except Exception as e:
            # ログ出力をしてエラーをスルー
            logger.error(f"Failed to process embedding for msg {message_id}: {e}")

    async def search(self, query: str, top_k: int = None, limit_months: int = None) -> list[dict]:
        """
        2段階ハイブリッド検索を実行する:
        第1段階: クエリを Sudachi で分割し、過去ログの chat_history.tokens を対象に BM25 で上位100件を抽出。
        第2段階: 抽出した100件のベクトルをSQLiteからロードし、クエリとのコサイン類似度を計算。
        """
        top_k = top_k or self.config.top_k
        limit_months = limit_months or self.config.search_range_months
        
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # 過去ログをフェッチ (直近 N ヶ月)
        tz = pytz.timezone(self.config.timezone)
        now = datetime.datetime.now(tz)
        cutoff_date = now - relativedelta(months=limit_months)
        cutoff_iso = cutoff_date.isoformat()

        cursor = await self.conn.execute("""
        SELECT message_id, timestamp, channel_id, channel_name, user_id, username, content, tokens, attachments
        FROM chat_history
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
        """, (cutoff_iso,))
        rows = await cursor.fetchall()
            
        if not rows:
            return []

        # 1. BM25 による絞り込み (最大上位100件)
        from rank_bm25 import BM25Okapi
        
        corpus = []
        doc_map = []
        for row in rows:
            tokens_str = row["tokens"]
            doc_tokens = tokens_str.split(" ") if tokens_str else []
            corpus.append(doc_tokens)
            doc_map.append(row)
            
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(query_tokens)
        
        candidates_limit = min(100, len(rows))
        top_indices = np.argsort(bm25_scores)[::-1][:candidates_limit]
        
        candidates = [doc_map[idx] for idx in top_indices]
        candidate_bm25_scores = [bm25_scores[idx] for idx in top_indices]

        # 2. クエリの Embedding ベクトルを取得
        try:
            query_vector = await get_embedding_with_retry(self.client, query, self.config.embedding_model)
            query_vector = np.array(query_vector, dtype=np.float32)
        except Exception as e:
            # Embedding に失敗した場合は純粋な BM25 の上位 K 件でフォールバック
            logger.warning(f"Embedding fail, falling back to pure BM25: {e}")
            results = []
            for doc, score in zip(candidates[:top_k], candidate_bm25_scores[:top_k]):
                results.append({
                    "message_id": doc["message_id"],
                    "timestamp": doc["timestamp"],
                    "channel_name": doc["channel_name"],
                    "username": doc["username"],
                    "content": doc["content"],
                    "score": float(score)
                })
            return results

        # 3. 候補者のベクトルを取得し、コサイン類似度を計算
        candidate_ids = [c["message_id"] for c in candidates]
        placeholders = ",".join("?" for _ in candidate_ids)
        
        cursor = await self.conn.execute(f"""
        SELECT message_id, vector
        FROM embeddings
        WHERE message_id IN ({placeholders})
        """, candidate_ids)
        vector_rows = await cursor.fetchall()
            
        vector_map = {}
        for vr in vector_rows:
            vector_map[vr["message_id"]] = np.frombuffer(vr["vector"], dtype=np.float32)

        final_scores = []
        max_bm25 = max(candidate_bm25_scores) if candidate_bm25_scores else 1.0
        min_bm25 = min(candidate_bm25_scores) if candidate_bm25_scores else 0.0
        bm25_diff = max_bm25 - min_bm25
        if bm25_diff == 0:
            bm25_diff = 1.0

        for doc, raw_bm25 in zip(candidates, candidate_bm25_scores):
            msg_id = doc["message_id"]
            norm_bm25 = (raw_bm25 - min_bm25) / bm25_diff
            
            if msg_id in vector_map:
                doc_vector = vector_map[msg_id]
                dot_product = np.dot(query_vector, doc_vector)
                norm_query = np.linalg.norm(query_vector)
                norm_doc = np.linalg.norm(doc_vector)
                cosine_sim = dot_product / (norm_query * norm_doc) if (norm_query * norm_doc) > 0 else 0.0
            else:
                cosine_sim = 0.0

            # ハイブリッドスコア算出
            score = (self.config.bm25_weight * norm_bm25) + (self.config.vector_weight * cosine_sim)
            final_scores.append((doc, score))

        # ソートして上位K件を抽出
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in final_scores[:top_k]:
            results.append({
                "message_id": doc["message_id"],
                "timestamp": doc["timestamp"],
                "channel_name": doc["channel_name"],
                "username": doc["username"],
                "content": doc["content"],
                "score": float(score)
            })
            
        return results
