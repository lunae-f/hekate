import os
import logging
from pathlib import Path

logger = logging.getLogger("persona")

class PersonaManager:
    """persona/ ディレクトリの Markdown ファイルからキャラクタープロファイルをロード・合成・キャッシュするクラス"""
    def __init__(self):
        self.persona_dir = Path(__file__).parent.parent / "persona"
        self._persona_cache = {
            "identity": "",
            "soul": "",
            "agents": ""
        }
        self._persona_mtimes = {
            "identity": 0.0,
            "soul": 0.0,
            "agents": 0.0
        }

    async def _load_persona(self) -> dict:
        """Markdownファイルを読み込む。更新時刻（mtime）をチェックし、更新があれば再読込（ホットリロード）"""
        files = {
            "identity": self.persona_dir / "IDENTITY.md",
            "soul": self.persona_dir / "SOUL.md",
            "agents": self.persona_dir / "AGENTS.md"
        }
        
        for key, filepath in files.items():
            try:
                if filepath.exists():
                    current_mtime = filepath.stat().st_mtime
                    if current_mtime > self._persona_mtimes[key]:
                        with open(filepath, "r", encoding="utf-8") as f:
                            self._persona_cache[key] = f.read().strip()
                        self._persona_mtimes[key] = current_mtime
                        logger.info(f"Loaded/Reloaded persona file: {filepath.name} (mtime={current_mtime})")
                else:
                    logger.warning(f"Persona file not found: {filepath}")
                    self._persona_cache[key] = ""
            except Exception as e:
                logger.error(f"Error loading persona file {filepath}: {e}")
                if not self._persona_cache[key]:
                    self._persona_cache[key] = ""
                    
        return self._persona_cache

    async def get_evaluator_instruction(self, base_instruction: str) -> str:
        """一次判定用の合成システムプロンプトを取得する"""
        persona = await self._load_persona()
        return f"{persona['identity']}\n\n{persona['soul']}\n\n{persona['agents']}\n\n{base_instruction}"

    async def get_generator_instruction(self, base_instruction: str) -> str:
        """返答生成用の合成システムプロンプトを取得する"""
        persona = await self._load_persona()
        return f"{persona['identity']}\n\n{persona['soul']}\n\n{persona['agents']}\n\n{base_instruction}"
