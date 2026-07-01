from openai import AsyncOpenAI
from config import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, SYSTEM_PROMPT

client = AsyncOpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)

conversation_history: list[dict] = []


async def generate_response(user_text: str) -> str:
    conversation_history.append({"role": "user", "content": user_text})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history[-20:],
    ]

    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=150,
        temperature=0.8,
    )

    reply = response.choices[0].message.content or "..."

    conversation_history.append({"role": "assistant", "content": reply})
    return reply


def reset_conversation() -> None:
    conversation_history.clear()
