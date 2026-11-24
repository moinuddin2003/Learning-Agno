from pathlib import Path
from agno.knowledge.reader.json_reader import JSONReader

from agno.db.sqlite import SqliteDb

from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from dotenv import load_dotenv
import os
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

json_path = Path(__file__).parent / "data.json"
reader = JSONReader()

LANCEDB_URI = "data/lancedb"

knowledge = Knowledge(
    vector_db=LanceDb(
        table_name="document",
        uri=LANCEDB_URI,
        search_type=SearchType.hybrid,
        embedder=OllamaEmbedder(id="nomic-embed-text", dimensions=768),
    ),
)

docs = reader.read(path=json_path)

print("Total Docs", len(docs))
for d in docs:
    print(d.content)
    print("-" * 30)
