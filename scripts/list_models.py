"""Lists the Gemini models your API key can use for chat: python scripts/list_models.py

The key is read from GEMINI_API_KEY; the output contains only model names.
"""
from google import genai

client = genai.Client()
for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(f"{model.name.removeprefix('models/'):40} {model.display_name}")
