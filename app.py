import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

load_dotenv()

app = Flask(__name__)

print("Загружаю модель...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("bank_docs")

# 1. Читаем TXT файл
with open("bank_doc.txt", "r", encoding="utf-8") as f:
    txt_text = f.read()

# 2. Читаем PDF файл
pdf_text = ""
reader = PdfReader("bank_faq.pdf")
for page in reader.pages:
    pdf_text += page.extract_text() + "\n"

# 3. Объединяем и делим на куски
all_text = txt_text + "\n\n" + pdf_text
chunks = [chunk.strip() for chunk in all_text.split("\n\n") if chunk.strip()]

# 4. Загружаем в векторную базу
print("Индексирую документы...")
embeddings = embedder.encode(chunks).tolist()
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"Загружено {len(chunks)} кусков из TXT и PDF. Сервер готов!")

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
        n_results=3
    )
    relevant_chunks = "\n".join(results["documents"][0])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""Ты вежливый помощник банка АО 'Тест Банк'.
Отвечай ТОЛЬКО на основе этих данных из документов банка:

{relevant_chunks}

Если ответа нет в документах — скажи 'Такой информации у меня нет, обратитесь к менеджеру'."""
            },
            {"role": "user", "content": question}
        ]
    )

    return jsonify({"answer": response.choices[0].message.content})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)