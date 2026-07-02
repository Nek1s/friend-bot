import os
import asyncio
import struct
import wave
import itertools
from concurrent.futures import ThreadPoolExecutor

import torch
import soundfile as sf

from config import (
    OUTPUT_DIR,
    QWEN_TTS_MODEL,
    TTS_LANGUAGE,
    REFERENCE_TEXT,
    VOICE_SAMPLES_DIR,
)

_pool = ThreadPoolExecutor(max_workers=1)

# Дефолтная частота дискретизации для тишины-заглушки, если модель не загружена.
_FALLBACK_SR = 24000


class TTSManager:
    def __init__(self) -> None:
        self.model = None
        self._loaded = False
        # ротация имён файлов: очередь может держать несколько предложений сразу
        self._seq = itertools.count()

    def load(self) -> None:
        if self._loaded:
            return
        print(f"TTSManager: загружаю Qwen3-TTS ({QWEN_TTS_MODEL}, первый запуск — ~2.5 ГБ весов)...")
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            QWEN_TTS_MODEL,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        self._loaded = True
        print("TTSManager: Qwen3-TTS модель загружена")

    def _find_speaker_wav(self) -> str | None:
        if not os.path.isdir(VOICE_SAMPLES_DIR):
            return None
        wavs = [f for f in os.listdir(VOICE_SAMPLES_DIR) if f.endswith(".wav")]
        if wavs:
            return os.path.join(VOICE_SAMPLES_DIR, wavs[0])
        return None

    def _generate_blocking(self, text: str, output_path: str, ref_audio: str) -> None:
        """Синхронный синтез — вызывается в отдельном потоке."""
        kwargs = dict(
            text=text,
            language=TTS_LANGUAGE,
            ref_audio=ref_audio,
        )
        if REFERENCE_TEXT:
            kwargs["ref_text"] = REFERENCE_TEXT
        wavs, sr = self.model.generate_voice_clone(**kwargs)
        sf.write(output_path, wavs[0], sr)

    async def generate(self, text: str) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # ротирующееся имя (mod 16), чтобы файлы в очереди не перетирались
        output_path = os.path.join(OUTPUT_DIR, f"tts_{next(self._seq) % 16}.wav")

        ref_audio = self._find_speaker_wav()

        if self.model is not None and ref_audio is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _pool,
                self._generate_blocking,
                text,
                output_path,
                ref_audio,
            )
        else:
            self._generate_silence(output_path)

        return output_path

    @staticmethod
    def _generate_silence(path: str, duration: float = 1.0) -> None:
        sample_rate = _FALLBACK_SR
        num_samples = int(sample_rate * duration)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack("<h", 0) * num_samples)


tts = TTSManager()
