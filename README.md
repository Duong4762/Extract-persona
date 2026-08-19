# Chạy pipeline persona

## 1. Thiết lập biến môi trường LLM

```powershell
$env:LLM_ENDPOINT = "http://203.113.152.4:7777/llm/v1/chat/completions"
$env:LLM_MODEL = "Qwen3-14B"
$env:LLM_AUTHORIZATION = "Basic <base64-credential>"
```

## 2. Chạy Amazon Reviews

```powershell
python amazon_extractor.py ingest
python amazon_extractor.py prepare
python amazon_extractor.py compact
python amazon_extractor.py extract
```

## 3. Chạy Vietnamese Reviews

```powershell
python vietnamese_reviews.py ingest
python vietnamese_reviews.py prepare
python vietnamese_reviews.py compact
python vietnamese_reviews.py extract
```
