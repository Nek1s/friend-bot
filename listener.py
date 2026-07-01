import io
import struct
import wave
import numpy as np
from faster_whisper import WhisperModel
from config import SILENCE_THRESHOLD, WHISPER_MODEL


class VoiceListener:
    def __init__(self, language: str = "ru") -> None:
        self.language = language
        self.whisper: WhisperModel | None = None
        self._whisper_loaded = False
        self.on_transcription = None

        self._buffer: list[bytes] = []
        self._silence_frames = 0
        self._silence_limit = int(SILENCE_THRESHOLD * 50)
        self._energy_threshold = 300
        self._debug_count = 0

    def load_whisper(self) -> None:
        if self._whisper_loaded:
            return
        print(f"VoiceListener: Faster-Whisper ({WHISPER_MODEL})... ~1.5 ГБ VRAM")
        self.whisper = WhisperModel(
            WHISPER_MODEL, device="cuda", compute_type="float16"
        )
        self._whisper_loaded = True
        print("VoiceListener: готов")

    def feed_pcm(self, pcm: bytes) -> str | None:
        energy = self._rms(pcm)
        is_speech = energy > self._energy_threshold

        self._debug_count += 1
        if self._debug_count % 50 == 1:
            buf = "Y" if self._buffer else "N"
            sp = "SPEECH" if is_speech else "silence"
            print(
                f"[VAD] frame={self._debug_count} energy={energy:.0f} "
                f"buf={buf} state={sp} pending={len(self._buffer)}"
            )

        if is_speech:
            self._buffer.append(pcm)
            self._silence_frames = 0
        elif self._buffer:
            self._buffer.append(pcm)
            self._silence_frames += 1
            if self._silence_frames >= self._silence_limit:
                print(f"[VAD] Silence threshold reached, transcribing...")
                return self._transcribe()

        return None

    def flush(self) -> str | None:
        if not self._buffer:
            return None
        return self._transcribe()

    def stop(self) -> None:
        self._buffer.clear()
        self._silence_frames = 0
        self.on_transcription = None

    def _transcribe(self) -> str | None:
        if not self._buffer or self.whisper is None:
            return None

        raw = b"".join(self._buffer)
        self._buffer.clear()
        self._silence_frames = 0

        mono = self._stereo_to_mono(raw)
        wav = self._to_wav(mono)

        print(f"[STT] Transcribing {len(wav)} bytes ({len(raw)} raw PCM)...")
        segments, _ = self.whisper.transcribe(
            wav, language=self.language, beam_size=5, vad_filter=True
        )
        text = " ".join(s.text for s in segments).strip()
        print(f"[STT] Result: '{text}'")
        return text or None

    @staticmethod
    def _rms(data: bytes) -> float:
        samples = struct.unpack(f"<{len(data) // 2}h", data)
        arr = np.array(samples, dtype=np.float32)
        return float(np.sqrt(np.mean(arr ** 2)))

    @staticmethod
    def _stereo_to_mono(data: bytes) -> bytes:
        samples = struct.unpack(f"<{len(data) // 2}h", data)
        mono = [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)]
        return struct.pack(f"<{len(mono)}h", *mono)

    @staticmethod
    def _to_wav(mono_pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(mono_pcm)
        return buf.getvalue()
