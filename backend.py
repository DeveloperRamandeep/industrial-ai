import pandas as pd
import re

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data.csv")

# Ensure correct types
df["Item"] = df["Item"].astype(str)
df["warehouse"] = df["warehouse"].astype(str)
df["category"] = df["category"].astype(str)

# =========================
# CLEAN QUERY
# =========================

def clean(text):
    return re.sub(r"[^a-zA-Z0-9 ]", "", text.lower()).strip()

# =========================
# FIND ITEM MATCH
# =========================

def find_item(query):
    query = query.lower()

    # exact match
    for _, row in df.iterrows():
        if row["Item"].lower() in query:
            return row

    # partial match
    matches = df[df["Item"].str.lower().str.contains(query, na=False)]

    if len(matches) > 0:
        return matches.iloc[0]

    return None

# =========================
# INTENT DETECTION
# =========================

def detect_intent(q):
    q = clean(q)

    if "price" in q or "cost" in q:
        return "price"

    if "stock" in q or "available" in q or "do you have" in q:
        return "stock"

    if "warehouse" in q or "where" in q:
        return "warehouse"

    if "delivery" in q or "lead" in q or "time" in q:
        return "lead"

    if "list" in q or "models" in q or "show" in q:
        return "list"

    if "cheap" in q:
        return "cheap"

    if "expensive" in q:
        return "expensive"

    return "general"

# =========================
# CHAT ENGINE
# =========================

def chat(query):
    intent = detect_intent(query)
    item = find_item(query)

    # ---------------- PRICE ----------------
    if intent == "price":
        if item is not None:
            return f"💰 The price of {item['Item']} is ₹{item['Price']}"
        return "❌ Item not found for price query."

    # ---------------- STOCK ----------------
    if intent == "stock":
        if "mumbai" in query.lower():
            items = df[df["warehouse"].str.lower() == "mumbai"]
            return "📦 Mumbai Stock Items:\n\n" + "\n".join(items["Item"].tolist())

        if item is not None:
            return f"📦 {item['Item']} → {item['stock_qty']} units ({item['Stock Status']})"

        return "❌ Stock information not found."

    # ---------------- WAREHOUSE ----------------
    if intent == "warehouse":
        for city in ["mumbai", "delhi", "bangalore", "hyderabad"]:
            if city in query.lower():
                items = df[df["warehouse"].str.lower() == city]
                return f"🏭 {city.title()} Warehouse Items:\n\n" + "\n".join(items["Item"].tolist())

        return "🏭 Warehouses: Mumbai, Delhi, Bangalore, Hyderabad"

    # ---------------- DELIVERY ----------------
    if intent == "lead":
        if item is not None:
            return f"⏱️ Delivery time for {item['Item']} is {item['Lead Time']} days"
        return "❌ Lead time not found."

    # ---------------- LIST ITEMS ----------------
    if intent == "list":
        return "📋 Sample Available Items:\n\n" + "\n".join(df["Item"].head(10).tolist())

    # ---------------- CHEAP ITEMS ----------------
    if intent == "cheap":
        cheap = df.sort_values("Price").head(5)
        return "💸 Cheapest Items:\n\n" + "\n".join(cheap["Item"].tolist())

    # ---------------- EXPENSIVE ITEMS ----------------
    if intent == "expensive":
        high = df.sort_values("Price", ascending=False).head(5)
        return "💰 Most Expensive Items:\n\n" + "\n".join(high["Item"].tolist())

    # ---------------- GENERAL HELP ----------------
    return """
🤖 I can help you with:

✔ Price of items
✔ Stock availability (Mumbai, Delhi, etc.)
✔ Warehouse location
✔ Delivery time / Lead time
✔ Available models
✔ Cheap & expensive items

Try asking:
- What is the price of Oil Seal Model 328?
- Is stock available in Mumbai?
- Show me cheapest items
- How long is delivery time?
"""

# =========================
# FASTAPI SUPPORT (OPTIONAL)
# =========================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Industrial Chatbot API Running"}

@app.post("/chat")
def chat_api(data: dict):
    message = data.get("message", "")
    reply = chat(message)
    return {"reply": reply}

# =========================
# LOCAL TEST
# =========================

if __name__ == "__main__":
    print("\n🔥 Industrial Chatbot Ready (Final Version)\n")

    while True:
        q = input("Ask: ")

        if q.lower() in ["exit", "quit"]:
            break

        print("\nAnswer:\n", chat(q))
