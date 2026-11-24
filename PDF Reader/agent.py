from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.recursive import RecursiveChunking
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
import os
from agno.agent import Agent
from agno.models.groq import Groq

from dotenv import load_dotenv
from agno.utils.log import set_log_level_to_debug


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


LANCEDB_URI = "data/lancedb"

BASE_DIR = os.path.dirname(__file__)

CONTENTS_DB_FILE = os.path.join(BASE_DIR, "knowledge_contents.db")


reader = PDFReader(
    chunking_strategy=RecursiveChunking(chunk_size=800, overlap=80),
    split_on_pages=True,
)

knowledge = Knowledge(
    vector_db=LanceDb(
        table_name="docs",
        uri=LANCEDB_URI,
        search_type=SearchType.hybrid,
        embedder=OllamaEmbedder(id="nomic-embed-text", dimensions=768),
    ),
    contents_db=SqliteDb(db_file=CONTENTS_DB_FILE),
    readers=reader,
)


agent = Agent(
    model=Groq(id="openai/gpt-oss-120b", api_key=api_key),
    search_knowledge=True,
    knowledge=knowledge,
    instructions=[
        "Always search the knowledge base before answering SQL questions.",
        "Answer only from the retrieved  information.",
        "If the Question is not in the knowledge base, say so.",
    ],
    markdown=True,
    debug_mode=1,
)


if __name__ == "__main__":
    set_log_level_to_debug()
    documents = reader.read(pdf=r"C:\Users\Xolva-01\Downloads\SQL_Questions.pdf")
    # (pdf=r"D:\AI_Training\Adeel\RecipeBook_WinterSoups_2022_Digital.pdf")
    text = "\n\n".join(d.content for d in documents if getattr(d, "content", None))
    print(text)
    knowledge.insert(
        path=r"C:\Users\Xolva-01\Downloads\SQL_Questions.pdf",
        skip_if_exists=True,
    )

    print("pdf process done")

    agent.print_response("Get the Top 3 Highest-Paid Employees (Google)", stream=True)
