import os
import aiofiles
import logging
from pathlib import Path
from src.config import config

ROOT_DIR = Path(__file__).parent.parent
MEM_DIR = ROOT_DIR / "data" / "memory"
LTM_PATH = MEM_DIR / "long_term_memory.md"

logger = logging.getLogger("retriever")

class MemoryRetriever:
    """短期、中期、長期記憶をカプセル化し、重複排除と文字数切り詰めを行うモジュール"""
    def __init__(self, qmd_engine):
        self.qmd = qmd_engine
        self.config = config

    async def get_context(self, current_query: str, recent_messages: list[dict]) -> str:
        """
        1. 短期記憶: 直近会話から message_id を抽出
        2. 中期記憶: QMDハイブリッド検索
        3. 重複排除: 検索結果から短期記憶にあるものを除外
        4. 長期記憶: 毎日要約される long_term_memory.md の読み込み
        5. 文字数制限: 合計文字数が config.max_context_chars を超えないよう切り詰め
        """
        logger.debug(f"Searching past memories for query: '{current_query}' JST...")

        # 1. 短期記憶のIDセットを作成
        recent_ids = {msg["message_id"] for msg in recent_messages}

        # 2. 中期記憶の検索
        search_results = await self.qmd.search(current_query)

        # 3. 重複排除
        filtered_results = []
        for res in search_results:
            if res["message_id"] not in recent_ids:
                filtered_results.append(res)

        # 4. 長期記憶 (要約メモリ) の読み込み (末尾から最大約2,000文字に制限)
        long_term_content = ""
        if LTM_PATH.exists():
            try:
                file_size = LTM_PATH.stat().st_size
                limit_chars = 2000
                # 日本語UTF-8のマルチバイトを考慮し、1文字最大3バイトとしてシーク位置を決定
                seek_offset = limit_chars * 3
                
                async with aiofiles.open(LTM_PATH, mode="r", encoding="utf-8") as f:
                    if file_size > seek_offset:
                        await f.seek(file_size - seek_offset)
                        raw_content = await f.read()
                        # マルチバイト破損を避けるため、シーク直後の最初の改行文字以降を取得
                        first_newline = raw_content.find("\n")
                        if first_newline != -1:
                            long_term_content = raw_content[first_newline + 1:]
                        else:
                            long_term_content = raw_content
                    else:
                        long_term_content = await f.read()
            except Exception as e:
                logger.warning(f"Failed to read long-term memory: {e}")

        # 5. 文字数制限付きでコンテキストをフォーマット
        context_parts = []
        char_count = 0
        max_chars = self.config.max_context_chars

        # 長期記憶を追加
        if long_term_content.strip():
            ltm_block = f"=== 長期記憶 (過去のやり取りの要約) ===\n{long_term_content.strip()}\n\n"
            context_parts.append(ltm_block)
            char_count += len(ltm_block)

        # 中期記憶 (ハイブリッド検索結果) を追加
        if filtered_results:
            mid_term_header = "=== 中期記憶 (過去の関連発言) ===\n"
            context_parts.append(mid_term_header)
            char_count += len(mid_term_header)
            
            for doc in filtered_results:
                doc_text = f"- [{doc['channel_name']}] {doc['username']}: {doc['content']}\n"
                if char_count + len(doc_text) > max_chars:
                    # 上限文字数を超える場合は切り捨て
                    break
                context_parts.append(doc_text)
                char_count += len(doc_text)

        context_str = "".join(context_parts) if context_parts else "過去の関連する文脈はありません。"

        num_duplicates = len(search_results) - len(filtered_results)
        logger.info(f"Retrieved {len(filtered_results)} mid-term memories. (Filtered {num_duplicates} duplicates in recent history. Total JST context size: {len(context_str)} chars)")

        return context_str
