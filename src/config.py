import os
import tomllib
from pathlib import Path

# プロジェクトルートからのパス
ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "config.toml"

class Config:
    def __init__(self):
        # デフォルト設定値
        self.timezone = "Asia/Tokyo"
        self.log_level = "INFO"
        
        self.evaluator_model = "gemini-3.1-flash-lite"
        self.generator_model = "gemini-3.5-flash"
        self.embedding_model = "gemini-embedding-2"
        self.model_cheap = "gemini-3.1-flash-lite"
        self.model_premium = "gemini-3.5-flash"
        self.temperature = 0.2
        self.enable_code_execution = False
        self.system_instruction = ""
        self.evaluator_instruction = ""
        self.generator_instruction = ""
        
        self.bm25_weight = 0.4
        self.vector_weight = 0.6
        self.top_k = 3
        self.recent_history_limit = 5
        self.search_range_months = 3
        self.max_context_chars = 4000
        
        self.trigger_keywords = []
        
        self.load()

    def load(self):
        if not CONFIG_PATH.exists():
            return
        
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
            
        system = data.get("system", {})
        self.timezone = system.get("timezone", self.timezone)
        self.log_level = system.get("log_level", self.log_level)
        
        gemini = data.get("gemini", {})
        self.evaluator_model = gemini.get("evaluator_model", self.evaluator_model)
        self.generator_model = gemini.get("generator_model", self.generator_model)
        self.embedding_model = gemini.get("embedding_model", self.embedding_model)
        self.model_cheap = gemini.get("model_cheap", self.model_cheap)
        self.model_premium = gemini.get("model_premium", self.model_premium)
        self.temperature = float(gemini.get("temperature", self.temperature))
        self.enable_code_execution = bool(gemini.get("enable_code_execution", self.enable_code_execution))
        self.system_instruction = gemini.get("system_instruction", self.system_instruction).strip()
        self.evaluator_instruction = gemini.get("evaluator_instruction", self.system_instruction).strip()
        self.generator_instruction = gemini.get("generator_instruction", self.system_instruction).strip()
        
        retrieval = data.get("retrieval", {})
        self.bm25_weight = float(retrieval.get("bm25_weight", self.bm25_weight))
        self.vector_weight = float(retrieval.get("vector_weight", self.vector_weight))
        self.top_k = int(retrieval.get("top_k", self.top_k))
        self.recent_history_limit = int(retrieval.get("recent_history_limit", self.recent_history_limit))
        self.search_range_months = int(retrieval.get("search_range_months", self.search_range_months))
        self.max_context_chars = int(retrieval.get("max_context_chars", self.max_context_chars))
        
        filt = data.get("filter", {})
        self.trigger_keywords = filt.get("trigger_keywords", self.trigger_keywords)

config = Config()
