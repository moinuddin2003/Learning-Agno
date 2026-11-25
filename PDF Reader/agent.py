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

PDF_PATH = r"C:\Users\Xolva-01\Downloads\ThaiRecipes.pdf"


LANCEDB_URI = "data/lancedb"

BASE_DIR = os.path.dirname(__file__)

CONTENTS_DB_FILE = os.path.join(BASE_DIR, "knowledge_contents.db")


reader = PDFReader(
    chunking_strategy=RecursiveChunking(chunk_size=500, overlap=75),
    split_on_pages=True,
)

def custom_retriever(query: str, agent=None, num_documents: int = 5, **kwargs):
    docs = knowledge.search(query=query, max_results= 5)
    return [doc.to_dict() for doc in docs]

knowledge = Knowledge(
    vector_db=LanceDb(
        table_name="docs",
        uri=LANCEDB_URI,
        search_type=SearchType.hybrid,
        embedder=OllamaEmbedder(id="nomic-embed-text", dimensions=768),
    ),
    contents_db=SqliteDb(db_file=CONTENTS_DB_FILE),
)


agent = Agent(
    model=Groq(id="openai/gpt-oss-120b", api_key=api_key),
    knowledge=knowledge,
    knowledge_retriever=custom_retriever,
    search_knowledge=True,
    add_knowledge_to_context=True,
    instructions=[
        "Always search the knowledge base before answering any questions.",
        "Answer only from the retrieved  information.",
        "If the Question is not in the knowledge base, say so.",
    ],
    markdown=True,
    debug_mode=1,
)


if __name__ == "__main__":
    set_log_level_to_debug()
    # documents = reader.read(PDF_PATH)
    # # (pdf=r"D:\AI_Training\Adeel\RecipeBook_WinterSoups_2022_Digital.pdf")
    # text = "\n\n".join(d.content for d in documents if getattr(d, "content", None))
    # for i, doc in enumerate(documents):
    #     content = getattr(doc, "content", "") or ""
    #     print(
    #         f"\n--- Chunk {i + 1} | Len: {len(content)} | Meta: {getattr(doc, 'meta_data', {})} ---"
    #     )
    # print(content[:200])
    knowledge.insert(
        name="Thai Recipes",
        path=PDF_PATH,
        reader=reader,
        skip_if_exists=True,
    )

    print("pdf process done")

    agent.print_response("Receipe for Pad Thai?", stream=True)
