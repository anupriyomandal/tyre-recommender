# Tyre Recommender CLI

A production-quality Python CLI application to recommend tyres for vehicles using semantic search over a CSV dataset. Answers are provided in natural language.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your OpenAI API key to `.env`.
3. Build the index from the CSV data:
   ```bash
   python src/main.py build-index --csv data/tyres.csv
   ```
4. Search for tyre recommendations:
   ```bash
   python src/main.py search "Tyres for Audi A3"
   ```
5. View index stats:
   ```bash
   python src/main.py stats
   ```

## Strict Context Behavior

The retriever now uses hybrid ranking:
- semantic search with FAISS to get candidates
- BM25 re-ranking to prioritize rows that actually match query keywords

If context is weak, the app returns exactly:
`Sorry, I don't know that`

You can tune strictness in `.env`:
- `BM25_MIN_SCORE` (default `0.05`)
- `OVERLAP_THRESHOLD_SHORT_QUERY` (default `0.5`, for <=2 tokens)
- `OVERLAP_THRESHOLD_MEDIUM_QUERY` (default `0.3`, for 3-5 tokens)
- `OVERLAP_THRESHOLD_LONG_QUERY` (default `0.2`, for >=6 tokens)
- `VECTOR_SIMILARITY_MIN` (default `0.58`, allows typo-tolerant semantic pass)
