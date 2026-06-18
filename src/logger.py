import sys
import logging
from src.config import config

class CustomFormatter(logging.Formatter):
    """警告レベル等の表記ゆれを統一したカスタムログフォーマッタ"""
    def format(self, record):
        levelname = record.levelname
        level = f"[{levelname}]"
        logger_name = f"[{record.name}]"
        return f"{level} {logger_name} {record.getMessage()}"

def setup_logging():
    """アプリケーション全体のロギング構成をセットアップする"""
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
