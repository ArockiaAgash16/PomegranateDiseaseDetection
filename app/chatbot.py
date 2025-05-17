import os
import warnings
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from openai import OpenAI
import sys

# ✅ Suppress TensorFlow INFO logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# ✅ Load BGE embedding model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

# ✅ Load Qdrant and embedding model
qdrant = QdrantClient(
    "..........",
    api_key="............"
)

collection_name = "pomegranate_disease_kb"

# ✅ OpenRouter API Setup for AI response generation
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="..................",
)

# ✅ Convert text into vector
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding / (np.linalg.norm(embedding) + 1e-10)

# ✅ Severity-aware Retrieval
def retrieve_by_disease_severity(disease, severity, top_k=5):
    query = f"{disease} {severity} treatment"
    vector = get_embedding(query).tolist()
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
        query_filter={"must": [
            {"key": "disease", "match": {"value": disease}},
            {"key": "severity", "match": {"value": severity}}
        ]}
    )

    chunks = []
    for r in results:
        text = r.payload["text"]
        source = r.payload.get("source", "")

        # ✅ Only keep useful sources (skip .pdf or "manual_kb")
        if source and not source.lower().endswith(".pdf") and source.lower() != "manual_kb":
            chunks.append(f"{text} (Source: {source})")
        else:
            chunks.append(text)

    return "\n".join(chunks)

# ✅ Main response generation logic
def generate_llm_response(disease, severity, query, chat_history):
    # ✅ Regex-based Crop Check
    # ✅ Reject queries explicitly referring to other crops or unrelated topics
    if (
        re.search(r"\bin\s+(banana|mango|grape|orange|tomato|chili|apple|fruit|crop|plant|tree)\b", query.lower()) or
        not re.search(r"\bpomegranate\b", query.lower()) and not any(d in query.lower() for d in ["alternaria", "anthracnose", "bacterial blight", "cercospora", "powdery mildew"])
    ):
        return "I am a pomegranate AI assistant. I can only assist with pomegranate-related diseases and treatments."

    chat_history.append({"user": query})

    retrieved_context = retrieve_by_disease_severity(disease, severity)
    cleaned_context = "\n".join([
        line for line in retrieved_context.split('\n')
        if not line.lower().startswith("source:")
    ])

    history_text = "\n".join([
        f"User: {chat['user']}\nAi Assistant: {chat.get('bot', '')}" for chat in chat_history
    ])

    # ✅ Expanded prompt to encourage longer, structured output
    prompt = (
        f"You are a pomegranate plant disease expert. Based on the context below, provide a clear and detailed treatment plan "
        f"for the detected disease and severity. Be specific about:\n"
        f"- Type and concentration of fungicides or bactericides\n"
        f"- Frequency and number of applications\n"
        f"- Organic or biological alternatives\n"
        f"- Preventive cultural practices\n"
        f"- Any additional advice for farmers\n\n"
        f"Your answer should be at least 4-5 lines long and must not mention document names.\n\n"
        f"When the query is totally unrelated to pomegranate disease or agriculture or any other disease remedy reply with I can only assist with pomegranate-related diseases and treatments.\n\n"
        f"Chat History:\n{history_text}\n\nContext:\n{retrieved_context}\n\nQuestion: {query}\nAnswer:"
    )

    llm_response = client.chat.completions.create(
        model="mistralai/mistral-small-3.1-24b-instruct:free",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict expert assistant who ONLY answers queries about pomegranate plant diseases, "
                    "treatments, pesticides, and severity-based solutions. "
                    "**Do not answer** any question that is not clearly about pomegranate. "
                    "If the query is unrelated, respond strictly with: "
                    "'I can only assist with pomegranate-related queries.'"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = llm_response.choices[0].message.content
    chat_history[-1]["bot"] = answer
    return answer

