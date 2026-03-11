import typer
from rich.console import Console
from rich.panel import Panel
from pathlib import Path

# Important: ensure src is in python path, handled implicitly when running as python src/main.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DEFAULT_CSV_PATH, FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL_NAME
from src.ingestion.csv_loader import load_csv
from src.utils.document_builder import build_document
from src.embeddings.embedding_model import EmbeddingModel
from src.indexing.faiss_indexer import FaissIndexer
from src.recommender.tyre_recommender import TyreRecommender
from src.utils.logger import get_logger

app = typer.Typer(help="Tyre Recommender CLI Application")
console = Console()
logger = get_logger(__name__)

@app.command()
def build_index(csv: Path = typer.Option(DEFAULT_CSV_PATH, help="Path to the tyres CSV file")):
    """
    Build the FAISS index from the tyre dataset.
    """
    console.print(f"[bold blue]Building index from {csv}...[/bold blue]")
    try:
        # Load CSV
        df = load_csv(csv)
        
        # Build documents
        console.print("[blue]Converting rows to documents...[/blue]")
        documents = df.apply(build_document, axis=1).tolist()
        metadata = df.to_dict('records')
        
        # Generate embeddings
        console.print(f"[blue]Generating embeddings using {EMBEDDING_MODEL_NAME}...[/blue]")
        embedding_model = EmbeddingModel()
        embeddings = embedding_model.encode_documents(documents)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        console.print(f"[blue]Building FAISS index (dimension: {dimension})...[/blue]")
        indexer = FaissIndexer(dimension=dimension)
        indexer.add_embeddings(embeddings, metadata)
        
        # Save index
        console.print(f"[blue]Saving index to {FAISS_INDEX_PATH}...[/blue]")
        indexer.save_index(FAISS_INDEX_PATH, METADATA_PATH)
        
        console.print("[bold green]Index built successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Failed to build index: {e}[/bold red]")
        raise typer.Exit(code=1)

@app.command()
def search(query: str):
    """
    Search for tyre recommendations using a natural language query.
    """
    console.print(f'[bold blue]Searching for "{query}"...[/bold blue]')
    try:
        recommender = TyreRecommender(FAISS_INDEX_PATH, METADATA_PATH)
        answer = recommender.recommend(query)
        
        console.print("\n------------------------------------------------")
        console.print(Panel(answer, title="Tyre Recommendation"))
        console.print("------------------------------------------------\n")
        
    except Exception as e:
        console.print(f"[bold red]Failed to process query: {e}[/bold red]")
        raise typer.Exit(code=1)

@app.command()
def stats():
    """
    Print statistics about the current index.
    """
    try:
        indexer = FaissIndexer(dimension=1536) # temp dim to allow load
        indexer.load_index(FAISS_INDEX_PATH, METADATA_PATH)
        
        num_vehicles = indexer.index.ntotal
        dimension = indexer.index.d
        index_type = "IndexFlatIP"
        
        console.print("\n[bold green]Index Statistics[/bold green]")
        console.print(f"Number of indexed vehicles: [cyan]{num_vehicles}[/cyan]")
        console.print(f"Embedding dimension: [cyan]{dimension}[/cyan]")
        console.print(f"Index type: [cyan]{index_type}[/cyan]\n")
        
    except FileNotFoundError:
        console.print("[bold red]Index not found. Please run build-index first.[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Failed to retrieve index stats: {e}[/bold red]")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
