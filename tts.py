import os
import struct
import wave
from config import OUTPUT_DIR, TTS_MODEL_NAME, VOICE_SAMPLES_DIR


class TTSManager:
    def __init__(self) -> None:
        self.model = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        # Место для загрузки XTTSv2:
        # from TTS.api import TTS
        # self.model = TTS(model_name=TTS_MODEL_NAME, progress_bar=False)
        # self.model.to("cuda")
        self._loaded = True
        print("TTSManager: модель не загружена (режим заглушки)")

    async def generate(self, text: str) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "tts_output.wav")

        if self.model is not None:
            # speaker_wav = os.path.join(VOICE_SAMPLES_DIR, "reference_neutral.wav")
            # self.model.tts_to_file(
            #     text=text,
            #     speaker_wav=speaker_wav,
            #     language="ru",
            #     file_path=output_path,
            # )
            pass
        else:
            self._generate_silence(output_path)

        return output_path

    @staticmethod
    def _generate_silence(path: str, duration: float = 1.0) -> None:
        sample_rate = 24000
        num_samples = int(sample_rate * duration)

        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack("<h", 0) * num_samples)


tts = TTSManager()
