import sys
import os
import warnings
import logging
from pathlib import Path

# Suppress warnings and standard logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Add the root directory to the python path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from rich.console import Console
from rich.panel import Panel
from src.config import FAISS_INDEX_PATH, METADATA_PATH
from src.recommender.tyre_recommender import TyreRecommender
from src.utils.logger import logger

logger.remove()

console = Console()

def main():
    console.print(Panel.fit("[bold blue]Tyre Recommender Assistant[/bold blue]\nType 'exit' or 'quit' to close the application."))
    
    try:
        # Initialize the recommender once
        with console.status("[blue]Loading FAISS index and initializing recommender...[/blue]"):
            recommender = TyreRecommender(FAISS_INDEX_PATH, METADATA_PATH)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize: {e}[/bold red]")
        console.print("[yellow]Hint: Did you run `python src/main.py build-index` first?[/yellow]")
        sys.exit(1)

    while True:
        try:
            query = console.input("\n[bold green]User>[/bold green] ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['exit', 'quit']:
                console.print("[blue]Goodbye![/blue]")
                break
                
            with console.status("[blue]Agent is thinking...[/blue]"):
                answer = recommender.recommend(query)
                
            console.print(f"\n[bold purple]Agent>[/bold purple] {answer}")
                
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            console.print("\n[blue]Goodbye![/blue]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    main()
