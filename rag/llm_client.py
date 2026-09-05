"""
Thin wrapper so the rest of the codebase never imports `openai` directly.
Switching PROVIDER in config.py from "openai" to "azure_openai" is the only
change needed to move the whole assistant onto Azure OpenAI.
"""
from __future__ import annotations

import config

_local_embedder = None
def get_client():
    if config.RAG_MODE == "free":
        from openai import OpenAI
        return OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)
    elif config.PROVIDER == "azure_openai":
        from openai import AzureOpenAI
        return AzureOpenAI(api_key=config.AZURE_OPENAI_API_KEY,
                            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                            api_version=config.AZURE_OPENAI_API_VERSION)
    else:
        from openai import OpenAI
        return OpenAI(api_key=config.OPENAI_API_KEY)



def embedding_model_name() -> str:
    return config.AZURE_EMBEDDING_DEPLOYMENT if config.PROVIDER == "azure_openai" else config.EMBEDDING_MODEL


def chat_model_name() -> str:
    if config.RAG_MODE == "free":
        return config.OPENROUTER_CHAT_MODEL
    elif config.PROVIDER == "azure_openai":
        return config.AZURE_CHAT_DEPLOYMENT
    else:
        return config.CHAT_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    if config.RAG_MODE == "free":
        embedder = _get_local_embedder()
        return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
    client = get_client()
    model = embedding_model_name()
    out = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(model=model, input=texts[i:i+100])
        out.extend([d.embedding for d in resp.data])
    return out


def chat_complete(messages, temperature=0.2, max_tokens=800) -> str:
    client = get_client()
    resp = client.chat.completions.create(model=chat_model_name(), messages=messages,
                                           temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content

def _get_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer
        _local_embedder = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    return _local_embedder

