import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/go/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")
# Для reasoning-моделей: "none" отключает «мысли» (быстрее до первого токена,
# напр. gemini-2.5-flash). Пусто — параметр не передаётся.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "")

FFMPEG_PATH = os.getenv(
    "FFMPEG_PATH",
    r"D:\Important\Projects\bot-rama\ffmpeg-8.1.2-full_build\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe",
)

VOICE_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "voice_samples")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

SYSTEM_PROMPT = (
    "Ты — ИИ-клон человека. Отвечай очень коротко, в разговорном стиле, "
    "как в живом диалоге: 1–2 коротких предложения, без маркдауна и длинных "
    "объяснений. Чем короче ответ — тем лучше."
)

SILENCE_THRESHOLD = 0.35
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")

# --- Qwen3-TTS (zero-shot voice clone) ---
QWEN_TTS_MODEL = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
# Qwen3-TTS ждёт имена языков строчными: 'russian', 'english', ... (или 'auto').
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "russian")
# Точный транскрипт того, что произнесено в voice_samples/reference_neutral.wav.
# Нужен Qwen3-TTS для качественного клонирования голоса.
REFERENCE_TEXT = os.getenv(
    "REFERENCE_TEXT",
    "Каждый человек обладает уникальным инструментом, который отражает его характер, "
    "настроение и жизненный опыт. Этот инструмент — наш голос. Когда мы говорим, "
    "звуковая волна мгновенно передает собеседнику тончайшие нюансы наших мыслей. "
    "Сегодня технологии шагнули далеко вперед. Идея воссоздать точную цифровую копию "
    "человеческого голоса больше не кажется сюжетом из фантастического романа. Это "
    "реальность, с которой мы работаем здесь и сейчас. Для того чтобы модель получилась "
    "качественной, важна каждая деталь: чистота записи, отсутствие эха и, конечно, "
    "естественность произношения.",
)
