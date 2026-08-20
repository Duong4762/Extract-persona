# Chạy pipeline persona

## 1. Thiết lập LLM

### Dùng endpoint local

```powershell
$env:LLM_PROVIDER = "local"
$env:LLM_ENDPOINT = "http://203.113.152.4:7777/llm/v1/chat/completions"
$env:LLM_MODEL = "Qwen3-14B"
$env:LLM_AUTHORIZATION = "Basic <base64-credential>"
```

### Dùng OpenRouter

```powershell
$env:LLM_PROVIDER = "openrouter"
$env:OPENROUTER_API_KEY = "<OPENROUTER_API_KEY>"
$env:OPENROUTER_MODEL = "google/gemma-4-31b-it:free"
```

Cả ba pipeline Amazon, Vietnamese Reviews và Stack Overflow sử dụng chung các biến trên.

## 2. Chạy Amazon Reviews

```powershell
python amazon_extractor.py ingest
python amazon_extractor.py prepare
python amazon_extractor.py compact
python amazon_extractor.py extract
python amazon_extractor.py stats
```

## 3. Chạy Vietnamese Reviews

```powershell
python vietnamese_reviews.py ingest
python vietnamese_reviews.py prepare
python vietnamese_reviews.py compact
python vietnamese_reviews.py extract
python vietnamese_reviews.py stats
```

## 4. Chạy Stack Overflow

Đặt các file `.jsonl` hoặc `.jsonl.gz` trong `data/stackoverflow`, sau đó chạy:

```powershell
python stackoverflow_extractor.py ingest
python stackoverflow_extractor.py prepare
python stackoverflow_extractor.py compact
python stackoverflow_extractor.py extract
python stackoverflow_extractor.py stats
```

## 5. Chạy VOZ

Đặt các file `.csv` trong `data/voz`, sau đó chạy:

```powershell
python voz_extractor.py ingest
python voz_extractor.py prepare
python voz_extractor.py compact
python voz_extractor.py extract
python voz_extractor.py stats
```
