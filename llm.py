from collections.abc import AsyncIterator

from openai import AsyncOpenAI
from config import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, SYSTEM_PROMPT

client = AsyncOpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)

conversation_history: list[dict] = []

_SENT_TERMINATORS = ".!?…"

# ВНИМАНИЕ: deepseek-v4-flash — reasoning-модель. Она тратит ~150–250 токенов на
# «размышление» (reasoning_content) ДО ответа (content). max_tokens считает и то,
# и другое, поэтому лимит должен быть с запасом — иначе reasoning съест весь бюджет
# и content окажется пустым (finish_reason=length).
MAX_TOKENS = 512


def _build_messages(user_text: str) -> list[dict]:
    conversation_history.append({"role": "user", "content": user_text})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history[-20:],
    ]


def _extract_sentences(buffer: str) -> tuple[list[str], str]:
    """Разбить накопленный буфер на завершённые предложения + остаток.

    Граница = терминатор (. ! ? …), за которым идёт пробел. Терминатор в самом
    конце буфера остаётся в остатке (вдруг следующий токен продолжит слово/число).
    """
    sentences: list[str] = []
    start = 0
    i = 0
    n = len(buffer)
    while i < n:
        if buffer[i] in _SENT_TERMINATORS:
            j = i + 1
            while j < n and buffer[j] in _SENT_TERMINATORS:
                j += 1
            if j < n and buffer[j].isspace():
                chunk = buffer[start:j].strip()
                if chunk:
                    sentences.append(chunk)
                start = j
                i = j
                continue
            if j >= n:
                break
        i += 1
    return sentences, buffer[start:]


async def generate_response(user_text: str) -> str:
    """Нестриминговый ответ целиком (для /ask и как фолбэк)."""
    messages = _build_messages(user_text)
    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.8,
            timeout=30,
        )
    except Exception as e:
        print(f"[LLM] ERROR: {e}")
        conversation_history.pop()
        return "..."

    reply = response.choices[0].message.content or "..."
    print(f"[LLM] {user_text!r} -> {reply!r}")
    conversation_history.append({"role": "assistant", "content": reply})
    return reply


async def stream_sentences(user_text: str) -> AsyncIterator[str]:
    """Стримить ответ LLM и отдавать по одному завершённому предложению.

    Позволяет начать TTS/воспроизведение первого предложения, пока модель ещё
    дописывает остальные. При ошибке стрима — фолбэк на цельный ответ.
    """
    messages = _build_messages(user_text)
    try:
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.8,
            timeout=30,
            stream=True,
        )
    except Exception as e:
        print(f"[LLM] stream ERROR: {e}")
        conversation_history.pop()  # снять добавленный user-ход, ответим фолбэком
        reply = await generate_response(user_text)
        if reply and reply != "...":
            yield reply
        return

    buffer = ""
    full = ""
    try:
        async for chunk in stream:
            if not chunk.choices:  # keep-alive/usage-чанк без выбора
                continue
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            buffer += delta
            full += delta
            sentences, buffer = _extract_sentences(buffer)
            for s in sentences:
                yield s
    except Exception as e:
        print(f"[LLM] stream read ERROR: {e}")

    tail = buffer.strip()
    if tail:
        yield tail
    full = full.strip()
    print(f"[LLM] {user_text!r} -> {full!r}")
    conversation_history.append({"role": "assistant", "content": full or "..."})


def reset_conversation() -> None:
    conversation_history.clear()
