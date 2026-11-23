from agno.knowledge.chunking.strategy import ChunkingStrategy
from agno.knowledge.document import Document
from agno.knowledge.chunking.recursive import RecursiveChunking


class SimpleTextChunker(ChunkingStrategy):
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def chunk(self, document: Document) -> list[Document]:
        chunks = []

        for start in range(0, len(document.content), self.chunk_size):
            chunk_text = document.content[start : start + self.chunk_size]
            chunk_number = len(chunks) + 1

            chunks.append(
                Document(
                    id=self._generate_chunk_id(
                        document,
                        chunk_number,
                        chunk_text,
                    ),
                    name=document.name,
                    content=chunk_text,
                    meta_data={
                        **(document.meta_data or {}),
                        "chunk_number": chunk_number,
                    },
                )
            )

        return chunks


# Your big string — stays only in memory
big_text = "Artificial Intelligence is changing how software developers build modern applications. Instead of writing rigid rules for every single case, developers now combine traditional code with language models that can read, summarize, and organize information.However, language models have limits on how much text they can process cleanly at one time. If you pass a massive file all at once, important details in the middle can get missed or response quality can drop.That is why chunking is used in real-world AI pipelines. By breaking a large document into smaller, well-structured pieces, the system can search faster, keep context clear, and pass only the most relevant parts to the model."

document = Document(
    name="my_text",
    content=big_text,
)

# chunker = SimpleTextChunker(chunk_size=500)

chunker = RecursiveChunking(
    chunk_size=200,
    overlap=50,
)


# This is your list of chunks
chunks = chunker.chunk(document)

print(f"Total chunks: {len(chunks)}\n")

for chunk in chunks:
    # print(f"--- Chunk {chunk.meta_data['chunk_number']} ---")
    print(chunk.content)
    print()


# This is your list of chunks
chunks = chunker.chunk(document)

print(f"Total chunks: {len(chunks)}\n")

for chunk in chunks:
    # print(f"--- Chunk {chunk.meta_data['chunk_number']} ---")
    print(chunk.content)
    print()
