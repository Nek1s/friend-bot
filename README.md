# friend-bot

Discord-бот — «ИИ-клон человека». Заходит в голосовой канал, слушает речь,
распознаёт её, отправляет в LLM, получает короткий ответ и озвучивает его
**клонированным голосом пользователя**, воспроизводя ответ обратно в канал.

```
Голос в Discord (Opus/PCM)
  → приём аудио (py-cord, PCMStreamSink)
  → VAD (энергетический порог)
  → STT (Faster-Whisper)
  → LLM (облако, OpenAI-совместимый API)
  → TTS (Qwen3-TTS, клон голоса)
  → воспроизведение через FFmpeg в голосовой канал
```

## Технологический стек

| Категория | Технология |
|-----------|-----------|
| Язык | Python 3.12 |
| Discord-фреймворк | py-cord (коммит `05cf65f`, голосовой приём) + `davey`, `PyNaCl` |
| STT | Faster-Whisper `medium` (CTranslate2, CUDA) |
| TTS | Qwen3-TTS `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (zero-shot клон голоса) |
| LLM | Облако, OpenAI-совместимый API (`openai` SDK) |
| ML-фронтенд | PyTorch / torchaudio `2.6.0+cu124` (CUDA 12.4) |
| Аудио | librosa, soundfile, numpy |
| Конфигурация | python-dotenv (`.env`) |
| Транскодирование | FFmpeg (внешний бинарь, путь в `FFMPEG_PATH`) |
| GPU | NVIDIA RTX 3080 Ti, 12 ГБ VRAM |

## Структура проекта

| Файл | Назначение |
|------|-----------|
| `main.py` | Бот, слэш-команды, склейка пайплайна, приём аудио и очередь воспроизведения |
| `listener.py` | `VoiceListener`: буфер PCM, энергетический VAD, STT, ресемпл 48k→16k |
| `llm.py` | Асинхронный клиент LLM, история диалога (20 сообщений), стриминг по предложениям |
| `tts.py` | `TTSManager`: синтез речи с клонированием голоса |
| `config.py` | Конфиг из `.env` (токены, модели, промпт, пути) |
| `voice_samples/` | Референс-сэмплы голоса для клонирования |
| `output/` | Последние сгенерированные аудио-ответы |

## Установка

```powershell
# 1. venv на Python 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 2. torch с CUDA 12.4 — ОТДЕЛЬНО и ПЕРЕД остальным
.\.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. остальные зависимости
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Также нужен **FFmpeg** — укажите путь к `ffmpeg.exe` в `FFMPEG_PATH`.

## Конфигурация

Создайте `.env` в корне проекта:

```dotenv
DISCORD_BOT_TOKEN=...

# LLM (OpenAI-совместимый API)
LLM_MODEL=gemini-2.5-flash
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=...
LLM_REASONING_EFFORT=none

# Опционально
FFMPEG_PATH=C:\path\to\ffmpeg.exe
WHISPER_MODEL=medium
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
TTS_LANGUAGE=russian
REFERENCE_TEXT=точный транскрипт voice_samples/reference_neutral.wav
```

`REFERENCE_TEXT` — **точный текст** того, что произнесено в референсном
аудио `voice_samples/reference_neutral.wav`: от него зависит качество
клонирования.

## Запуск

```powershell
.\.venv\Scripts\python.exe main.py
```

Веса Qwen3-TTS (~2.5 ГБ) скачиваются при первом запуске.

## Команды бота

| Команда | Описание |
|---------|----------|
| `/ping` | Проверка работоспособности |
| `/join` | Подключиться к голосовому каналу |
| `/leave` | Отключиться от голосового канала |
| `/listen` | Начать слушать |
| `/talk` | Подключиться + слушать одной командой |
| `/stoplisten` | Остановить прослушивание |
| `/chat <text>` | Отправить текст — ответ голосом |
| `/ask <text>` | Отправить текст — ответ текстом |
| `/reset` | Сбросить историю диалога |
| `/play <filepath>` | Проиграть wav-файл в голосовой канал |

## Как это работает

- **VAD** — энергетический порог (`SILENCE_THRESHOLD = 0.35`): тишина
  завершает реплику, после чего запускается STT.
- **Стриминг ответа** — LLM стримится, ответ режется на предложения,
  каждое предложение сразу идёт в TTS и в очередь воспроизведения. Первое
  предложение начинает играть, пока следующие ещё генерируются.
- **Barge-in** — новый входящий запрос очищает очередь и обрывает
  текущее воспроизведение.
- **Системный промпт** просит 1–2 коротких предложения: TTS — самое
  тяжёлое звено пайплайна, длина ответа напрямую влияет на задержку.

## Оптимизация задержки

| Этап | Время (прогретые модели) |
|------|--------------------------|
| STT (16 с аудио) | ~0.9 с |
| LLM до 1-го токена | ~0.8–1.1 с (Gemini, `reasoning_effort=none`) |
| TTS (RTF ~2.6) | 3 с речи → ~8 с |

Отклонённые/отложенные варианты: flash-attn (нет CUDA Toolkit/MSVC на
машине), потоковая генерация TTS (кандидат на следующий этап).

## Зависимости

См. [`requirements.txt`](requirements.txt). Ключевые: `torch 2.6.0+cu124`,
`qwen-tts 0.1.1`, `py-cord` (git), `davey`, `PyNaCl`, `faster-whisper 1.2.1`,
`openai 2.44.0`, `librosa 0.11.0`, `soundfile 0.14.0`, `numpy`, `python-dotenv`.

## Заметки

- `roadmap.md` описывает исходный план с Coqui XTTSv2 — историческая
  справка, TTS уже заменён на Qwen3-TTS.
- Подробная документация и траблшутинг — в [`CLAUDE.md`](CLAUDE.md).
