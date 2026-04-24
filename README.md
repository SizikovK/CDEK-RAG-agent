# CDEK RAG Agent

Сервис чат-бота по базе знаний программы стажировки **CdekStart**.

## Что делает программа
- Поднимает FastAPI API с одним endpoint: `POST /chat`.
- Использует LangGraph + RAG (SQLiteVSS + embeddings) для поиска релевантного контекста.
- Поддерживает историю диалога(до выхода из программы).

## Основной конвейер
```
Запрос -> поиск по базе знаний -> LLM -> tool (запросы в бд по необходимости) -> ответ
```

## Технический стек
- Git
- Docker Compose
- Python
- FastAPI
- LangGraph / LangChain
- SQLiteVSS

## Конфигурация LLM
Отредактируйте `.env` под свои нужды:

### OpenAI

```env
PROVIDER=openai
LLM=gpt-4o-mini
API_KEY=your_key
BASE_URL=
TEMPERATURE=0

KNOWLEDGE_BASE=./data
VECTOR_DB=./database/vss.db
EMBENDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

### Ollama

```env
PROVIDER=ollama
LLM=llama3.1
API_KEY=
BASE_URL=http://host.docker.internal:11434
TEMPERATURE=0

KNOWLEDGE_BASE=./data
VECTOR_DB=./database/vss.db
EMBENDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

Если хотите использовать другие модели, установите соответствующие зависимости

## Установка и запуск
```bash
docker compose up --build
```

После запуска API доступен на `http://localhost:8000`.

## API
### `POST /chat`
- `Content-Type: application/json`

Тело запроса:
```json
{
  "message": "Какие условия участия для Германии?",
  "reset_history": false
}
```

Ответ:
```json
{
  "answer": "..."
}
```

## Быстрая проверка

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"Какие условия участия для Германии?"}'
```

Со сбросом истории
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"Привет","reset_history":true}'
```