import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from config import OUTPUT_DIR, TTS_MODEL_NAME, VOICE_SAMPLES_DIR

# PyTorch 2.6+ default: weights_only=True — TTS needs False
torch.serialization.add_safe_globals([])
_original_load = torch.load


def _load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _load

_pool = ThreadPoolExecutor(max_workers=1)


class TTSManager:
    def __init__(self) -> None:
        self.model = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        print("TTSManager: загружаю XTTSv2 (первый запуск — ~3 ГБ весов)...")
        from TTS.api import TTS

        self.model = TTS(model_name=TTS_MODEL_NAME, progress_bar=False)
        self.model.to("cuda")
        self._loaded = True
        print("TTSManager: XTTSv2 модель загружена")

    def _find_speaker_wav(self) -> str | None:
        if not os.path.isdir(VOICE_SAMPLES_DIR):
            return None
        wavs = [f for f in os.listdir(VOICE_SAMPLES_DIR) if f.endswith(".wav")]
        if wavs:
            return os.path.join(VOICE_SAMPLES_DIR, wavs[0])
        return None

    async def generate(self, text: str) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "tts_output.wav")

        speaker_wav = self._find_speaker_wav()

        if self.model is not None and speaker_wav is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _pool,
                lambda: self.model.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    language="ru",
                    file_path=output_path,
                ),
            )
        else:
            self._generate_silence(output_path)

        return output_path

    @staticmethod
    def _generate_silence(path: str, duration: float = 1.0) -> None:
        import struct
        import wave

        sample_rate = 24000
        num_samples = int(sample_rate * duration)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack("<h", 0) * num_samples)


tts = TTSManager()
