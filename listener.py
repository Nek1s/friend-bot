import io
import struct
import wave
import numpy as np
from faster_whisper import WhisperModel
from config import SILENCE_THRESHOLD, WHISPER_MODEL

try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False


class VoiceListener:
    def __init__(self, language: str = "ru") -> None:
        self.language = language
        self.whisper: WhisperModel | None = None
        self._whisper_loaded = False
        self.on_transcription = None

        self._buffer: list[bytes] = []
        self._silence_frames = 0
        self._speech_streak = 0
        self._silence_limit = int(SILENCE_THRESHOLD * 50)
        self._energy_threshold = 800
        self._max_frames = 400  # 8 sec — force flush
        self._debug_count = 0
        self.busy = False
        self._triggered = False  # guard against duplicate triggers

    def is_busy(self) -> bool:
        return self.busy

    def load_whisper(self) -> None:
        if self._whisper_loaded:
            return
        print(f"VoiceListener: Faster-Whisper ({WHISPER_MODEL})... ~1.5 ГБ VRAM")
        self.whisper = WhisperModel(
            WHISPER_MODEL, device="cuda", compute_type="float16"
        )
        self._whisper_loaded = True
        print("VoiceListener: готов")

    def feed_pcm(self, pcm: bytes) -> bool:
        """Returns True when transcription should be triggered (async by caller)."""
        if self.busy or self._triggered:
            return False
        energy = self._rms(pcm)
        is_speech = energy > self._energy_threshold

        self._debug_count += 1
        if self._debug_count % 50 == 1:
            buf = "Y" if self._buffer else "N"
            sp = "SPEECH" if is_speech else "silence"
            print(
                f"[VAD] frame={self._debug_count} energy={energy:.0f} "
                f"buf={buf} state={sp} silence={self._silence_frames} streak={self._speech_streak}"
            )

        if is_speech:
            self._speech_streak += 1
            if self._speech_streak >= 3:
                self._silence_frames = 0
            self._buffer.append(pcm)
        else:
            self._speech_streak = 0
            if self._buffer:
                self._buffer.append(pcm)
                self._silence_frames += 1
                if self._silence_frames >= self._silence_limit:
                    print("[VAD] Silence threshold reached, transcribing...")
                    self._triggered = True
                    return True

        if len(self._buffer) >= self._max_frames:
            print(f"[VAD] Max buffer ({self._max_frames} frames) reached, force-flushing...")
            self._triggered = True
            return True

        return False

    def flush(self) -> str | None:
        if not self._buffer:
            return None
        return self._transcribe()

    def transcribe_blocking(self) -> str | None:
        """Run transcription; should be called in a background thread."""
        return self._transcribe()

    def has_pending(self) -> bool:
        """Check if buffer has data ready to transcribe (without running it)."""
        return bool(self._buffer) and not self.busy

    def stop(self) -> None:
        self._buffer.clear()
        self._silence_frames = 0
        self.on_transcription = None

    def _transcribe(self) -> str | None:
        if not self._buffer or self.whisper is None:
            return None

        self.busy = True
        self._triggered = False
        raw = b"".join(self._buffer)
        self._buffer.clear()
        self._silence_frames = 0
        self._speech_streak = 0

        mono = self._stereo_to_mono(raw)

        print(f"[STT] Transcribing {len(mono)} bytes ({len(raw)} raw PCM)...")
        try:
            segments, _ = self.whisper.transcribe(
                self._to_numpy_16k(mono),
                language=self.language,
                beam_size=1,
                vad_filter=False,
            )
            text = " ".join(s.text for s in segments).strip()
            print(f"[STT] Result: '{text}'")
            return text or None
        finally:
            self.busy = False

    def _to_numpy_16k(self, mono_pcm: bytes, sr_in: int = 48000) -> np.ndarray:
        """Convert mono PCM bytes to float32 numpy array at 16kHz."""
        arr = np.frombuffer(mono_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if _HAS_LIBROSA and sr_in != 16000:
            arr = librosa.resample(arr, orig_sr=sr_in, target_sr=16000)
        elif sr_in != 16000:
            step = sr_in / 16000
            idx = np.arange(0, len(arr), step).astype(int)
            arr = arr[idx]
        return arr

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

    
