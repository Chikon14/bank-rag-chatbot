from groq import Groq

# Читаем банковский документ
with open("bank_doc.txt", "r", encoding="utf-8") as f:
    document = f.read()

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("=" * 50)
print("Добро пожаловать в чат-бот Тест Банка!")
print("Для выхода напишите 'выход'")
print("=" * 50)

# История разговора
history = []

while True:
    question = input("\nВы: ")
    
    if question.lower() == "выход":
        print("До свидания!")
        break
    
    # Добавляем вопрос в историю
    history.append({"role": "user", "content": question})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""Ты вежливый помощник банка АО 'Тест Банк'.
Отвечай ТОЛЬКО на основе этого документа:

{document}

Если ответа нет в документе — скажи 'Такой информации у меня нет, обратитесь к менеджеру'."""
            }
        ] + history
    )
    
    answer = response.choices[0].message.content
    
    # Добавляем ответ в историю
    history.append({"role": "assistant", "content": answer})
    
    print(f"\nБанк: {answer}")