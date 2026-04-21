import pandas as pd
import faiss
import numpy as np
import re
import random
import sqlite3
from sentence_transformers import SentenceTransformer
from ollama import Client

# =========================
# DATA CONFIG
# =========================

random.seed(42)

CATEGORIES = {
    "Seals": ["Oil Seal", "Pipe Seal", "Rubber Seal"],
    "Gaskets": ["Engine Gasket", "Pump Gasket"],
    "Valves": ["Ball Valve", "Gate Valve", "Check Valve"],
    "Pumps": ["Water Pump", "Hydraulic Pump"]
}

WAREHOUSES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad"]

def get_stock_status():
    r = random.random()
    if r < 0.7:
        return "available"
    elif r < 0.9:
        return "limited"
    else:
        return "out_of_stock"

def get_stock_qty(status):
    if status == "available":
        return random.randint(30, 120)
    elif status == "limited":
        return random.randint(5, 25)
    else:
        return 0

def make_name(base):
    return f"{base} Model {random.randint(100,999)}"

# =========================
# GENERATE DATA
# =========================

def generate_data(n=200):
    data = []

    for i in range(n):
        category = random.choice(list(CATEGORIES.keys()))
        base_item = random.choice(CATEGORIES[category])

        status = get_stock_status()
        stock = get_stock_qty(status)

        data.append({
            "part_id": f"P-{1000+i}",
            "Item": make_name(base_item),
            "category": category,
            "stock_qty": stock,
            "warehouse": random.choice(WAREHOUSES),
            "Lead Time": random.choice([1, 3, 5, 7, 10]),
            "Price": round(random.uniform(500, 50000), 2),
            "Stock Status": status
        })

    return pd.DataFrame(data)

df = generate_data(200)

# =========================
# SAVE CSV + SQLITE
# =========================

df.to_csv("data.csv", index=False)

conn = sqlite3.connect("parts.db")
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS parts")

cur.execute("""
CREATE TABLE parts (
    part_id TEXT,
    Item TEXT,
    category TEXT,
    stock_qty INTEGER,
    warehouse TEXT,
    Lead_Time INTEGER,
    Price REAL,
    Stock_Status TEXT
)
""")

for _, r in df.iterrows():
    cur.execute("INSERT INTO parts VALUES (?,?,?,?,?,?,?,?)", (
        r["part_id"], r["Item"], r["category"],
        r["stock_qty"], r["warehouse"],
        r["Lead Time"], r["Price"], r["Stock Status"]
    ))

conn.commit()

# =========================
# FAISS SETUP
# =========================

docs = []
rows = []

for _, row in df.iterrows():
    docs.append(f"""
Item: {row['Item']}
Category: {row['category']}
Status: {row['Stock Status']}
Price: {row['Price']}
Stock: {row['stock_qty']}
Warehouse: {row['warehouse']}
""".strip())

    rows.append(row)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.array(model.encode(docs)).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# =========================
# SEARCH
# =========================

def search(query, top_k=3):
    q = model.encode([query])
    q = np.array(q).astype("float32")

    _, idx = index.search(q, top_k)
    return [rows[i] for i in idx[0]]

# =========================
# CONTEXT
# =========================

def build_context(results):
    blocks = []
    for r in results:
        blocks.append(f"""
Item: {r['Item']}
Status: {r['Stock Status']}
Price: {r['Price']}
Stock: {r['stock_qty']}
Warehouse: {r['warehouse']}
""".strip())
    return "\n\n".join(blocks)

# =========================
# CLEAN
# =========================

def clean_output(text):
    return re.sub(r"STRICT.*", "", text, flags=re.IGNORECASE).strip()

# =========================
# OLLAMA CLIENT (FIXED)
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
            {"role": "system", "content": "Answer clearly using context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return clean_output(res["message"]["content"])

# =========================
# CHAT
# =========================

def chat(query):
    results = search(query)
    context = build_context(results)
    return ask_llm(context, query)

# =========================
# RUN LOOP
# =========================

if __name__ == "__main__":
    print("\n🔥 System Ready\n")

    while True:
        q = input("Ask: ")
        if q.lower() in ["exit", "quit"]:
            break

        print("\nAnswer:\n", chat(q))