import asyncio
import io
import struct
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import webrtcvad
from faster_whisper import WhisperModel
from config import SILENCE_THRESHOLD, WHISPER_MODEL

_pool = ThreadPoolExecutor(max_workers=1)

SAMPLE_RATE = 48000
FRAME_DURATION_MS = 20
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2 * 2
VAD_SAMPLE_RATE = 16000


class VoiceListener:
    def __init__(
        self,
        language: str = "ru",
        silence_threshold: float = SILENCE_THRESHOLD,
    ) -> None:
        self.language = language
        self.silence_threshold = silence_threshold
        self.vad = webrtcvad.Vad(2)
        self.whisper: WhisperModel | None = None
        self._whisper_loaded = False

        self.audio_buffer: list[bytes] = []
        self.silence_duration: float = 0.0
        self.is_speaking = False
        self.on_transcription: callable | None = None

    def load_whisper(self) -> None:
        if self._whisper_loaded:
            return
        print(
            f"VoiceListener: загружаю Faster-Whisper ({WHISPER_MODEL})... "
            f"~1.5 ГБ VRAM"
        )
        self.whisper = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type="float16",
        )
        self._whisper_loaded = True
        print("VoiceListener: Faster-Whisper загружен")

    def feed_pcm(self, data: bytes) -> str | None:
        """Process incoming PCM audio, return transcribed text or None."""
        if not data:
            return None

        is_speech = self._is_speech(data)

        if is_speech:
            self.audio_buffer.append(data)
            self.silence_duration = 0.0
            if not self.is_speaking:
                self.is_speaking = True
                self.audio_buffer.clear()
                self.audio_buffer.append(data)
        else:
            if self.is_speaking:
                self.audio_buffer.append(data)
                self.silence_duration += FRAME_DURATION_MS / 1000

                if self.silence_duration >= self.silence_threshold:
                    self.is_speaking = False
                    self.silence_duration = 0.0
                    return self._transcribe()

        return None

    def flush(self) -> str | None:
        if not self.audio_buffer:
            return None
        self.is_speaking = False
        self.silence_duration = 0.0
        return self._transcribe()

    def stop(self) -> None:
        self.audio_buffer.clear()
        self.is_speaking = False
        self.silence_duration = 0.0
        self.on_transcription = None

    def _is_speech(self, data: bytes) -> bool:
        mono_data = self._stereo_to_mono(data)
        resampled = self._resample_48k_to_16k(mono_data, len(mono_data) // 2)
        vad_frame = resampled[: int(VAD_SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2]
        return self.vad.is_speech(vad_frame, VAD_SAMPLE_RATE)

    def _transcribe(self) -> str | None:
        if not self.audio_buffer or self.whisper is None:
            return None

        wav_bytes = self._pack_wav()
        self.audio_buffer.clear()

        segments, _ = self.whisper.transcribe(
            wav_bytes,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text or None

    def _pack_wav(self) -> bytes:
        raw = b"".join(self.audio_buffer)
        mono = self._stereo_to_mono(raw)
        resampled = self._resample_48k_to_16k(mono, len(mono) // 2)

        buf = io.BytesIO()
        with wave.open(buf, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(resampled)
        return buf.getvalue()

    @staticmethod
    def _stereo_to_mono(data: bytes) -> bytes:
        samples = struct.unpack(f"<{len(data) // 2}h", data)
        mono = [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)]
        return struct.pack(f"<{len(mono)}h", *mono)

    @staticmethod
    def _resample_48k_to_16k(data: bytes, num_samples: int) -> bytes:
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        ratio = 16000 / 48000
        new_len = int(len(arr) * ratio)
        resampled = np.interp(
            np.linspace(0, len(arr) - 1, new_len), np.arange(len(arr)), arr
        )
        resampled = (resampled * 32767).astype(np.int16)
        return resampled.tobytes()


class AudioSink:
    def __init__(self, listener: VoiceListener) -> None:
        self.listener = listener
        self.loop = asyncio.get_event_loop()

    def write(self, data: bytes) -> None:
        result = self.listener.feed_pcm(data)
        if result and self.listener.on_transcription:
            asyncio.run_coroutine_threadsafe(
                self.listener.on_transcription(result), self.loop
            )
