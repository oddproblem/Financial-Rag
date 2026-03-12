import sys
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings

print("Got env and imports")
emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
print("Got model")
try:
    res = emb.embed_query("test")
    print(f"Success! Vector length: {len(res)}")
except Exception as e:
    print(f"Error: {e}")
