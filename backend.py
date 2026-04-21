import pandas as pd
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from ollama import Client

# =========================
# LOAD DATA FROM CSV ONLY
# =========================

df = pd.read_csv("data.csv")

# Ensure required columns exist
required_cols = ["Item", "category", "stock_qty", "warehouse", "Lead Time", "Price", "Stock Status"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column in data.csv: {col}")

# =========================
# LAZY MODEL LOADING
# =========================

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return _model

# =========================
# SIMPLE SEARCH (NO FAISS)
# =========================

def search(query, top_k=3):
    model = get_model()

    items = df["Item"].astype(str).tolist()

    item_vecs = model.encode(items, show_progress_bar=False)
    query_vec = model.encode([query], show_progress_bar=False)

    scores = np.dot(item_vecs, query_vec.T).flatten()

    top_idx = scores.argsort()[-top_k:][::-1]

    return df.iloc[top_idx].to_dict(orient="records")

# =========================
# CONTEXT BUILDER
# =========================

def build_context(results):
    return "\n\n".join([
        f"""
Item: {r['Item']}
Category: {r['category']}
Status: {r['Stock Status']}
Price: {r['Price']}
Stock: {r['stock_qty']}
Warehouse: {r['warehouse']}
Lead Time: {r['Lead Time']}
""".strip()
        for r in results
    ])

# =========================
# CLEAN OUTPUT
# =========================

def clean_output(text):
    return re.sub(r"STRICT.*", "", text, flags=re.IGNORECASE).strip()

# =========================
# OLLAMA CLIENT
# =========================

API_KEY = "e1419a15f08844e4be64e32a4acb712c.Xd-U5ZIZowsiftPGq38pg3G8"

client = Client(
    host="https://ollama.com",
    headers={
        "Authorization": f"Bearer {API_KEY}"
    }
)

def ask_llm(context, query):
    res = client.chat(
        model="gpt-oss:120b",
        messages=[
            {
                "role": "system",
                "content": "You are an industrial assistant. Answer using context clearly and simply."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )

    return clean_output(res["message"]["content"])

# =========================
# MAIN CHAT FUNCTION
# =========================

def chat(query):
    results = search(query)
    context = build_context(results)
    return ask_llm(context, query)

# =========================
# LOCAL TEST
# =========================

if __name__ == "__main__":
    print("\n🔥 CSV-Based Backend Running\n")

    while True:
        q = input("Ask: ")
        if q.lower() in ["exit", "quit"]:
            break

        print("\nAnswer:\n", chat(q))
