from pathlib import Path
from agno.agent import Agent
from agno.knowledge.chunking.fixed import FixedSizeChunking
from agno.knowledge.document import Document
from agno.models.groq import Groq
from agno.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
folder = Path(__file__).parent

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    doc = Document(name="text", content=text)
    chunks = FixedSizeChunking(chunk_size=chunk_size, overlap=overlap).chunk(doc)
    return [c.content for c in chunks]


chunk_agent = Agent(
    name="Chunker",
    model=Groq(id="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY")),
    tools=[chunk_text],
    instructions="Use chunk_text tool to split the given text into chunks.",
    debug_mode=True,
    markdown=True,
)

big_text = "Artificial Intelligence is changing how software developers build modern applications. Instead of writing rigid rules for every single case, developers now combine traditional code with language models that can read, summarize, and organize information. However, language models have limits on how much text they can process cleanly at one time. If you pass a massive file all at once, important details in the middle can get missed or response quality can drop."

chunk_agent.print_response(f"Chunk this text:\n{big_text}", stream=True)
