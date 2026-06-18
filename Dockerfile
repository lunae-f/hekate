FROM python:3.11-slim

# システムの依存関係を最小限に設定 (タイムゾーンデータ等)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先に依存パッケージファイルをコピーしてキャッシュを有効化
COPY requirements.txt .

# pipキャッシュマウントを使用して高速ビルド
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# アプリケーションコードをコピー
COPY config.toml .
COPY src/ ./src/

# データ永続化ディレクトリを作成
RUN mkdir -p data/memory data/index

# 環境変数の設定 (PYTHONPATHを追加してsrcモジュールをインポート可能にする)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.main"]
