# Hybrid Document Search: TF-IDF + BERT

A notebook comparing three document-retrieval approaches and combining them into a hybrid ranker:

1. **TF-IDF** search (scikit-learn `TfidfVectorizer` + cosine similarity), with the fitted vectorizer and matrix pickled for reuse.
2. **BERT** semantic search (`bert-base-uncased` embeddings from Hugging Face Transformers, cosine similarity).
3. **Hybrid** search that blends TF-IDF and BERT similarity scores.

The notebook also reads text/PDF documents (PyPDF2, `chardet` for encoding detection), compares result quality and query latency across the three models, and includes a small timing benchmark.

## Run

Designed for Google Colab. Put your documents (`.txt` / `.pdf`) in a Drive folder and point the notebook's document path (`/content/drive/MyDrive/documents/`) to it.

```bash
pip install torch transformers scikit-learn chardet PyPDF2
```

## Files

- `Project2Research.ipynb` - implementation and experiments.

## Report

- [Project report (PDF)](docs/ML_Project_Report.pdf)
