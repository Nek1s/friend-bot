# CLAUDE.md — friend-bot

> Документация проекта + план миграции TTS с Coqui XTTSv2 на **Qwen3-TTS 0.6B-Base**.
> Язык общения в этом проекте — русский.

## 1. Что это за проект

**friend-bot** — Discord-бот, «ИИ-клон человека». Он заходит в голосовой канал,
слушает речь, распознаёт её, отправляет в облачную LLM, получает короткий ответ,
озвучивает его **клонированным голосом пользователя** и проигрывает обратно в канал.

Пайплайн реального времени:

```
Голос в Discord (Opus/PCM)
  → приём аудио (py-cord voice receive, PCMStreamSink)
  → VAD (энергетический порог в listener.py)
  → STT (Faster-Whisper medium, ru)
  → LLM (облако, OpenAI-совместимый API)
  → TTS (клон голоса)          ← ЗДЕСЬ МЕНЯЕМ XTTSv2 на Qwen3-TTS
  → воспроизведение через FFmpeg в голосовой канал
```

LLM — облачная (`opencode.ai/zen`), поэтому GPU расходуется только на STT + TTS.

## 2. Структура файлов

| Файл | Назначение |
|------|-----------|
| `main.py` | Бот, слэш-команды, склейка пайплайна, `PCMStreamSink` для приёма аудио |
| `listener.py` | `VoiceListener`: буфер PCM, энергетический VAD, Faster-Whisper STT, ресемпл 48k→16k |
| `llm.py` | Асинхронный клиент LLM (OpenAI SDK), история диалога (последние 20 сообщений) |
| `tts.py` | **`TTSManager`** — синтез речи. **Файл переписывается под Qwen3-TTS** |
| `config.py` | Конфиг из `.env` (токены, модели, промпт, пути) |
| `voice_samples/reference_neutral.wav` | Референс-сэмпл голоса для клонирования (~7 МБ) |
| `output/tts_output.wav` | Последний сгенерированный аудио-ответ |
| `roadmap.md` | Исходный роадмап (историческая справка, там ещё описан XTTS) |

## 3. Команды бота (slash)

`/ping`, `/join`, `/leave`, `/listen`, `/talk` (join+listen одной командой),
`/stoplisten`, `/chat <text>` (ответ голосом), `/ask <text>` (ответ текстом),
`/reset` (сброс истории), `/play <filepath>`.

## 4. Окружение и стек

- **ОС**: Windows 10, PowerShell (основной shell), также доступен Bash (Git Bash).
- **GPU**: NVIDIA RTX 3080 Ti, **12 ГБ VRAM**, CUDA 12.4 (torch `+cu124`).
- **Python**: сейчас 3.11.3 → **мигрируем на 3.12.0** (`py -3.12`, установлен в
  `C:\Users\User\AppData\Local\Programs\Python\Python312`).
- **FFmpeg**: путь задан в `config.py` / `.env` (`FFMPEG_PATH`).
- **Ключевые библиотеки**: `torch/torchaudio 2.6.0+cu124`, `py-cord` (voice receive,
  конкретный git-коммит), `PyNaCl`, `faster-whisper 1.2.1`, `openai 2.44.0`,
  `librosa 0.11.0`, `soundfile 0.14.0`.
- **`.env`** (не в git): `DISCORD_BOT_TOKEN`, `LLM_MODEL`, `LLM_BASE_URL`,
  `LLM_API_KEY`; опционально `FFMPEG_PATH`, `WHISPER_MODEL`.

Запуск: `.venv\Scripts\python.exe main.py`.

## 5. Бюджет VRAM (после миграции)

| Компонент | VRAM |
|-----------|------|
| Faster-Whisper `medium` (float16) | ~1.5–2 ГБ |
| Qwen3-TTS 0.6B-Base (bf16) | ~1.5–2 ГБ |
| **Итого** | **~3.5–4 ГБ из 12 ГБ** — с большим запасом |

---

# 6. ПЛАН МИГРАЦИИ: Coqui XTTSv2 → Qwen3-TTS 0.6B-Base

## 6.1. Принятые решения

| Вопрос | Решение |
|--------|---------|
| Вариант модели | **`Qwen/Qwen3-TTS-12Hz-0.6B-Base`** (zero-shot клон из референса) |
| Транскрипт референса (`ref_text`) | Пользователь даёт **точный текст** → в `config.py` поле `REFERENCE_TEXT` |
| Окружение | **Пересоздать venv на Python 3.12** (чистое окружение, как советует Qwen) |
| Порядок | Написать план → **сразу внедрять** |

## 6.2. Факты о Qwen3-TTS (проверено, open-source ~янв 2026, Alibaba Qwen)

- **pip-пакет**: `pip install -U qwen-tts`
- **Модель**: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (~2.5 ГБ, скачается при первом запуске)
- **Загрузка**:
  ```python
  from qwen_tts import Qwen3TTSModel
  import torch
  model = Qwen3TTSModel.from_pretrained(
      "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
      device_map="cuda:0",
      dtype=torch.bfloat16,
  )
  ```
- **Синтез с клонированием голоса**:
  ```python
  wavs, sr = model.generate_voice_clone(
      text="текст для озвучки",
      language="russian",          # СТРОЧНЫМИ; supported: auto/chinese/english/french/
                                   # german/italian/japanese/korean/portuguese/russian/spanish
      ref_audio="voice_samples/reference_neutral.wav",
      ref_text="точный транскрипт референса",
  )
  import soundfile as sf
  sf.write("output/tts_output.wav", wavs[0], sr)   # sr отдаёт сама модель
  ```
- Поддержка **русского** — да (одна из 10 языков: zh, en, ja, ko, de, fr, **ru**, pt, es, it).
- CUDA, bfloat16, стриминг (латентность ~97 мс) — стриминг оставляем на будущее.
- Рекомендуется Python 3.12; опционально FlashAttention 2 для экономии памяти.

## 6.3. Отличия от XTTSv2, которые влияют на код

1. **Нужен транскрипт референса** (`ref_text`). У XTTS хватало только `speaker_wav`.
   → добавляем `REFERENCE_TEXT` в конфиг (пользователь заполнит).
2. **Язык строкой строчными**: `"russian"` (не `"ru"`, не `"Russian"` — регистр важен,
   иначе значение не в списке поддерживаемых и модель уйдёт в авто-детект).
3. **Другой API вызова**: `generate_voice_clone(...)` вместо `tts_to_file(...)`.
4. **Запись через soundfile**: модель возвращает `(wavs, sr)`, пишем `sf.write`.
5. **Не нужен monkeypatch `torch.load`** (это был костыль XTTS под PyTorch 2.6
   `weights_only`). Qwen грузится через HF transformers — патч удаляем.
6. **Частота дискретизации** больше не фиксированные 24000; берём `sr` из модели.
   `discord.FFmpegPCMAudio` определяет её сам, так что проигрывание не ломается.

## 6.4. Шаги внедрения

### Шаг 1 — Пересоздать venv на Python 3.12
```powershell
# из корня проекта (d:\Important\Projects\bot-rama\friend-bot)
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### Шаг 2 — Поставить torch (CUDA 12.4) и остальные зависимости
```powershell
.\.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -U qwen-tts
.\.venv\Scripts\python.exe -m pip install "py-cord @ git+https://github.com/Pycord-Development/pycord@05cf65fa6d81567a8a4347ac961a093981554137"
.\.venv\Scripts\python.exe -m pip install PyNaCl faster-whisper openai librosa soundfile python-dotenv numpy
```
> Coqui `TTS` **не ставим** — вместе с ним уходит гора транзитивных зависимостей
> (spacy, gruut, trainer, encodec…) и пин `transformers==4.49.0`. `qwen-tts`
> подтянет свою совместимую версию `transformers`.

### Шаг 3 — Переписать `tts.py`
- Убрать блок monkeypatch `torch.load` и импорт `from TTS.api import TTS`.
- `load()`: грузить `Qwen3TTSModel.from_pretrained(QWEN_TTS_MODEL, device_map="cuda:0", dtype=torch.bfloat16)`.
- `generate(text)`: вызвать `generate_voice_clone(text, language=TTS_LANGUAGE, ref_audio=<reference_neutral.wav>, ref_text=REFERENCE_TEXT)`, записать `sf.write(output_path, wavs[0], sr)` в executor (не блокировать event loop).
- Сохранить `_find_speaker_wav()` (реф-аудио) и fallback `_generate_silence()`.

### Шаг 4 — Обновить `config.py`
- Удалить `TTS_MODEL_NAME` (Coqui).
- Добавить:
  ```python
  QWEN_TTS_MODEL = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
  TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "Russian")
  REFERENCE_TEXT = os.getenv("REFERENCE_TEXT", "<точный транскрипт reference_neutral.wav>")
  ```

### Шаг 5 — Проверка
1. `.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"` → `True`.
2. `.\.venv\Scripts\python.exe -c "from qwen_tts import Qwen3TTSModel"` — импорт без ошибок.
3. Мини-скрипт: загрузить модель + `generate_voice_clone` на короткой фразе → проверить `output/*.wav`.
4. Запустить бота, `/talk`, сказать фразу — убедиться, что ответ звучит голосом клона.

## 6.5. Риски и на что смотреть

- **Скачивание ~2.5 ГБ** весов при первом `load()` — первый запуск дольше.
- **numpy 2.x vs 1.26**: свежий venv может поставить numpy 2.x; проверить совместимость
  с `faster-whisper`/`librosa` (обычно ок, но если что — закрепить `numpy<2`).
- **Python 3.12.0** — «нулевой» релиз; если попадётся баг колёс, есть 3.11 как запасной.
- **`ref_text` должен соответствовать содержимому `reference_neutral.wav`** — иначе клон хуже.
- **Длина референса**: текущий `reference_neutral.wav` = **37 сек** (stereo, 48 кГц), а Qwen
  рекомендует **3–10 сек**. Стратегия: сначала тест с полным файлом; если ошибка/плохое
  качество — обрезать аудио до ~10 сек и синхронно урезать `REFERENCE_TEXT` до первых
  предложений, реально звучащих в этом отрезке.
- **FlashAttention 2** — опционально; на Windows ставится тяжело, без неё модель работает.

## 6.6. Обновить после миграции
- Зафиксирован `requirements.txt` (курированный) для воспроизводимости. ✅
- В `.env.example` (если появится) добавить `QWEN_TTS_MODEL`, `TTS_LANGUAGE`, `REFERENCE_TEXT`.
- `roadmap.md` описывает XTTS — оставить как историю или пометить, что TTS заменён.

---

# 7. Статус миграции (проверено локально)

Дата: 2026-07-02. Миграция выполнена и проверена на уровне компонентов.

| Проверка | Результат |
|----------|-----------|
| venv пересоздан на Python 3.12.0 | ✅ |
| torch 2.6.0+cu124, `torch.cuda.is_available()` | ✅ `True` |
| `qwen-tts==0.1.1` установлен, API совпал с кодом | ✅ |
| Qwen3-TTS 0.6B-Base загрузка + генерация речи (ru) | ✅ `output/tts_output.wav`, ~6 сек на тест-фразу |
| Полный 37-сек референс отработал без ошибки | ✅ (обрезать не понадобилось) |
| Faster-Whisper `medium` на CUDA + транскрипция | ✅ |
| Импорт всего приложения (`main`/`listener`/`tts`/`llm`) | ✅ |
| 10 слэш-команд регистрируются | ✅ |
| **Живой тест в Discord (голос → ответ голосом)** | ⏳ выполняет пользователь |

Запуск бота: `.\.venv\Scripts\python.exe main.py`, затем в Discord `/talk` и говорить.
Старый venv сохранён как `.venv_xtts_backup` (можно удалить после подтверждения работы).

# 8. Траблшутинг Qwen3-TTS

- **`davey is required for voice support`** — voice-бэкенд py-cord. Ставится отдельно:
  `pip install davey` (в requirements есть). Голосовой приём/воспроизведение без него не работает.
- **Регистр языка**: Qwen ждёт `language` **строчными** (`russian`, `english`, …). С заглавной
  (`Russian`) значение выпадает из списка и модель уходит в авто-детект → кривой темп/повторы.
  Поддерживаемые: `auto, chinese, english, french, german, italian, japanese, korean,
  portuguese, russian, spanish`.
- **`SoX could not be found` при импорте** — питон-обёртка `sox` (транзитивная зависимость
  qwen-tts) ищет CLI-бинарь SoX. Для клонирования голоса **не требуется** (генерация работает),
  предупреждение можно игнорировать. При желании убрать — поставить SoX в PATH.
- **`flash-attn is not installed`** — только скорость; модель работает на «ручном» PyTorch.
- **`hf_xet` not installed** — только скорость загрузки весов с HF; опционально `pip install hf_xet`.
