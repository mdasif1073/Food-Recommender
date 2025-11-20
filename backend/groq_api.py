import os
import logging
import requests
from typing import List, Dict

logger = logging.getLogger("groq_api")

GROQ_URL = os.getenv("GROQ_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" # Using a fast and capable model

# --- System Persona ---
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a friendly, enthusiastic, and expert food recommender for Coimbatore, India. "
        "Your tone is warm and conversational. You provide a single, excellent food recommendation based on the data provided. "
        "You don't just list facts; you weave them into a natural, persuasive paragraph. "
        "Always end by asking for feedback in a friendly way."
    )
}

def groq_chat(prompt: str, history: List[Dict[str, str]] | None = None, temperature: float = 0.7) -> str:
    """
    Generic Groq chat wrapper with a system persona for conversational responses.
    """
    if not GROQ_URL or not GROQ_API_KEY:
        return "LLM is currently unavailable (missing API credentials)."

    messages = [SYSTEM_PROMPT]
    if history:
        # Add only the last few turns of history to keep context
        for m in history[-4:]: # Limit to last 2 user/bot turns
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role in ("bot", "assistant"):
                messages.append({"role": "assistant", "content": content})

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 250,
    }

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=20 # Increased timeout for generation
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {}).get("content", "")
        return msg or "Sorry, I couldn't generate a proper response."
    except requests.exceptions.Timeout:
        logger.warning("Groq API timed out.")
        return "Sorry, the recommendation is taking too long to generate. Please try again."
    except Exception as e:
        logger.warning(f"Groq API call failed: {e}")
        return "My thinking cap isn't working right now! I can't generate a conversational response."