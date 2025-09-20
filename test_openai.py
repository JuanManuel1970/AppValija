import os
from dotenv import load_dotenv
load_dotenv(override=True)

import openai
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    models = client.models.list()
    print("✅ Conexión OK. Ejemplo de modelo:", models.data[0].id if models.data else "sin datos")
except Exception as e:
    print("❌ Error OpenAI:", e)
