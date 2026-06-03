import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)

# Загружаем модель и базу при старте
print("Загружаю модель...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("bank_docs")

with open("bank_doc.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
embeddings = embedder.encode(chunks).tolist()
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"Загружено {len(chunks)} кусков. Сервер готов!")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    
    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    relevant_chunks = "\n".join(results["documents"][0])
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""Ты помощник банка АО 'Тест Банк'.
Отвечай ТОЛЬКО на основе этих данных:

{relevant_chunks}

Если ответа нет — скажи 'Такой информации у меня нет'."""
            },
            {"role": "user", "content": question}
        ]
    )
    
    return jsonify({"answer": response.choices[0].message.content})

if __name__ == "__main__":
    app.run(debug=True)