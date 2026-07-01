import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/go/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")

FFMPEG_PATH = os.getenv(
    "FFMPEG_PATH",
    r"D:\Important\Projects\bot-rama\ffmpeg-8.1.2-full_build\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe",
)

VOICE_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "voice_samples")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

SYSTEM_PROMPT = (
    "Ты — ИИ-клон человека. Отвечай коротко, емко, в разговорном стиле, "
    "как в живом диалоге. Без маркдауна и длинных объяснений. "
    "Максимум 2–3 предложения."
)

SILENCE_THRESHOLD = 1.5
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
TTS_MODEL_NAME = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
