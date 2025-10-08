"""Demo script: process a small text document, index it, and run a query.

Run from repository root:
    python scripts/demo_ui_flow.py

This mirrors what the Streamlit UI does, but runs headless for quick verification.
"""
import os
import sys
from pathlib import Path
import tempfile
import json

# Ensure `src` is on sys.path so local package imports work when running this script
repo_root = Path(__file__).resolve().parents[1]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from ingestion.document_processor import DocumentProcessor, DocumentChunk  # type: ignore
from extraction.vector_store import EmbeddingManager, ChromaVectorStore  # type: ignore
from query.query_engine import DocumentQueryEngine  # type: ignore


def write_sample(path: Path):
    text = (
        "This is a demo document about machine learning.\n"
        "Machine learning is a field of artificial intelligence that uses statistical techniques\n"
        "to enable computers to learn from data. Common algorithms include linear regression,\n"
        "decision trees, and neural networks.\n"
    )
    path.write_text(text, encoding="utf-8")
    return str(path)


def main():
    repo = Path(__file__).resolve().parents[1]

    # Create a small sample text file
    sample_dir = repo / "data" / "demo"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_file = sample_dir / "demo_doc.txt"
    sample_path = write_sample(sample_file)

    print(f"Sample document written to: {sample_path}")

    # Process document (handle txt specially since DocumentProcessor may not support txt)
    chunks = []
    file_ext = Path(sample_path).suffix.lower()
    if file_ext == ".txt":
        text = Path(sample_path).read_text(encoding="utf-8")
        # simple chunking: split into paragraphs
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        for i, p in enumerate(paras):
            chunk = DocumentChunk(
                content=p,
                metadata={"file_name": Path(sample_path).name, "file_path": sample_path},
                chunk_id=f"demo-{i}",
                document_id="demo-doc",
                page_number=None,
                section_title=None,
                chunk_type="text"
            )
            chunks.append(chunk)
        print(f"Created {len(chunks)} chunks from txt file")
    else:
        processor = DocumentProcessor()
        processed = processor.process_document(sample_path)
        chunks = processed.chunks
        print(f"Processed document, chunks: {len(chunks)}")

    # Embed and add to vector store
    emb_mgr = EmbeddingManager()
    chunk_embeddings = emb_mgr.embed_chunks(chunks)

    chroma = ChromaVectorStore()
    chroma.add_chunks(chunk_embeddings)
    print("Indexed chunks into ChromaDB")

    # Create query engine and query
    hf_model_env = os.getenv("HF_MODEL")
    hf_use_local_env = os.getenv("HF_USE_LOCAL")

    fallback_model = "gpt2"
    chosen_model = hf_model_env or fallback_model

    heavy_models = {"tiiuae/falcon-7b-instruct", "falcon-7b-instruct", "llama-2", "llama2"}
    if chosen_model.lower() in heavy_models:
        print(f"[demo] Model '{chosen_model}' is large; overriding to '{fallback_model}' for demo run.")
        chosen_model = fallback_model

    use_local = (hf_use_local_env or "true").lower() == "true"

    overrides = {
        "provider": "hf",
        "hf": {
            "model": chosen_model,
            "use_local": use_local,
            "max_tokens": int(os.getenv("HF_MAX_TOKENS", "512")),
            "temperature": float(os.getenv("HF_TEMPERATURE", "0.3"))
        }
    }

    print("[demo] Using HF model overrides:", json.dumps(overrides["hf"], indent=2))

    engine = DocumentQueryEngine(overrides=overrides)
    result = engine.query("What is machine learning?", n_results=3)

    print("\nQuery result:\n", result.answer)


if __name__ == "__main__":
    main()
