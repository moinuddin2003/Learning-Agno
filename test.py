from agno.knowledge.chunking.strategy import ChunkingStrategy
from agno.knowledge.document import Document
 
 
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
big_text = "Your very large text goes here... " * 1000
 
document = Document(
    name="my_text",
    content=big_text,
)
 
chunker = SimpleTextChunker(chunk_size=500)
 
# This is your list of chunks
chunks = chunker.chunk(document)
 
print(f"Total chunks: {len(chunks)}\n")
 
for chunk in chunks:
    print(f"--- Chunk {chunk.meta_data['chunk_number']} ---")
    print(chunk.content)
    print()