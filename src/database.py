from pathlib import Path
import os
from threading import Lock

from config import EMBENDING_MODEL, KNOWLEDGE_BASE, VECTOR_DB
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)
from langchain_community.vectorstores import SQLiteVSS


_EMBEDDING: SentenceTransformerEmbeddings | None = None
_INDEX_LOCK = Lock()

def get_embedding() -> SentenceTransformerEmbeddings:
    """Кэширует модель эмбеддингов в процессе"""
    global _EMBEDDING
    if _EMBEDDING is None:
        _EMBEDDING = SentenceTransformerEmbeddings(model_name=EMBENDING_MODEL)
    return _EMBEDDING


def return_all_files() -> list[dict[str, str]]:
    """Читает файлы из базы знаний"""
    path = Path(KNOWLEDGE_BASE)
    texts: list[dict[str, str]] = []

    if not path.exists():
        return texts

    for filename in path.rglob("*"):
        if not filename.is_file():
            continue

        try:
            content = filename.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        texts.append(
            {
                "source": str(filename),
                "content": content,
            }
        )

    return texts


def open_db(db_file: str = VECTOR_DB) -> SQLiteVSS:
    """Открывает БД без индексации"""
    Path(db_file).parent.mkdir(parents=True, exist_ok=True)

    db = SQLiteVSS(
        table="state_union",
        connection=None,
        db_file=db_file,
        embedding=get_embedding(),
    )
    return db


def _is_table_empty(db: SQLiteVSS) -> bool:
    row = db._connection.execute(f"SELECT COUNT(*) AS count FROM {db._table}").fetchone()
    return row["count"] == 0

def _index_all_files(db: SQLiteVSS) -> int:
    files = return_all_files()
    texts = [item["content"] for item in files]
    metadatas = [{"source": item["source"]} for item in files]

    if not texts:
        return 0

    db.add_texts(texts=texts, metadatas=metadatas)
    return len(texts)


def index_init() -> None:
    """Инициализация индекса: выполняется при первом запросе"""
    with _INDEX_LOCK:
        db = open_db()
        try:
            if _is_table_empty(db):
                _index_all_files(db)
        finally:
            db._connection.close()


def index_db(db: SQLiteVSS) -> int:
    """Принудительно добавляет файлы в индекс"""
    return _index_all_files(db)


def remove_db(db_file: str = VECTOR_DB) -> None:
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Файл {db_file} удален")
    else:
        print("Файл не найден")
