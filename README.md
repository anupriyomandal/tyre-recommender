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
