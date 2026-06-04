import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from groq import Groq
from pypdf import PdfReader

load_dotenv()

app = Flask(__name__)

# Читаем TXT файл
with open("bank_doc.txt", "r", encoding="utf-8") as f:
    txt_text = f.read()

# Читаем PDF файл
pdf_text = ""
reader = PdfReader("bank_faq.pdf")
for page in reader.pages:
    pdf_text += page.extract_text() + "\n"

# Объединяем все документы
all_documents = txt_text + "\n\n" + pdf_text

print("Документы загружены. Сервер готов!")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""Ты вежливый помощник банка АО 'Тест Банк'.
Отвечай ТОЛЬКО на основе этих документов банка:

{all_documents}

Если ответа нет в документах — скажи 'Такой информации у меня нет, обратитесь к менеджеру'."""
            },
            {"role": "user", "content": question}
        ]
    )

    return jsonify({"answer": response.choices[0].message.content})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)