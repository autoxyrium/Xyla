"""
XYLA Main Runner
================
Entry point for Xyla on Raspberry Pi.
Ties together:
  - Vision engine (camera + CV)
  - AI engine (Claude)
  - Voice input/output (mic + speaker)
  - Mirror UI (served via local HTTP)

Run with:
    python src/vision/xyla_main.py --api-key sk-ant-...
    
    # Or set env var:
    export ANTHROPIC_API_KEY=sk-ant-...
    python src/vision/xyla_main.py

Author: Xyla Project
"""

import argparse
import os
import sys
import time
import threading
import logging
import json
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.vision.vision_engine import XylaVisionEngine
from src.vision.ai_engine import XylaAI

log = logging.getLogger("XYLA")
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s %(message)s'
)


# ═══════════════════════════════════════════════════════════
# VOICE I/O
# ═══════════════════════════════════════════════════════════

class VoiceIO:
    """
    Handles speech-to-text (Whisper) and text-to-speech (Piper).
    Both run fully offline on RPi — no internet needed for voice.
    """

    def __init__(self):
        self._whisper_model = None
        self._tts_available = False
        self._setup()

    def _setup(self):
        """Try to load Whisper and Piper."""
        # Whisper STT
        try:
            import whisper
            log.info("Loading Whisper tiny model for STT (offline)...")
            self._whisper_model = whisper.load_model("tiny")  # ~39MB, fast on RPi5
            log.info("Whisper loaded")
        except ImportError:
            log.warning("Whisper not installed. Run: pip install openai-whisper")
        except Exception as e:
            log.warning(f"Whisper load failed: {e}")

        # Piper TTS — check if binary exists
        piper_path = os.path.join(os.path.dirname(__file__), "../../models/piper/piper")
        if os.path.exists(piper_path):
            self._tts_available = True
            self._piper_path = piper_path
            self._piper_model = os.path.join(
                os.path.dirname(__file__),
                "../../models/piper/en_US-lessac-medium.onnx"
            )
            log.info("Piper TTS available (offline)")
        else:
            log.warning("Piper TTS not found. Run: python src/vision/download_models.py")

    def listen(self, duration: float = 5.0) -> str:
        """
        Record from mic and transcribe.
        Returns transcribed text, or empty string if failed.
        """
        if not self._whisper_model:
            log.warning("Whisper not available — using text input only")
            return ""

        import tempfile
        import sounddevice as sd
        import soundfile as sf
        import numpy as np

        log.info(f"Listening for {duration}s...")
        sample_rate = 16000
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            result = self._whisper_model.transcribe(f.name, language="en")
            os.unlink(f.name)

        text = result.get("text", "").strip()
        log.info(f"Heard: {text}")
        return text

    def speak(self, text: str):
        """
        Speak text aloud using Piper TTS.
        Falls back to espeak if Piper not available.
        """
        # Clean text for TTS
        clean = text.replace("✨", "").replace("💄", "").replace("👗", "").strip()

        if self._tts_available:
            self._speak_piper(clean)
        else:
            self._speak_espeak(clean)

    def _speak_piper(self, text: str):
        """Use Piper neural TTS (natural-sounding, offline)."""
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                out_path = f.name

            cmd = [
                self._piper_path,
                "--model", self._piper_model,
                "--output_file", out_path
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            proc.communicate(input=text.encode())

            # Play the audio
            subprocess.run(["aplay", out_path], check=True, capture_output=True)
            os.unlink(out_path)
        except Exception as e:
            log.warning(f"Piper TTS failed: {e}")
            self._speak_espeak(text)

    def _speak_espeak(self, text: str):
        """Fallback: use espeak (robotic but always available on RPi)."""
        try:
            subprocess.run(
                ["espeak-ng", "-s", "150", "-v", "en-gb", text],
                capture_output=True, timeout=30
            )
        except Exception as e:
            log.warning(f"espeak failed: {e}")


# ═══════════════════════════════════════════════════════════
# WAKE WORD DETECTION
# ═══════════════════════════════════════════════════════════

class WakeWordDetector:
    """
    Detects "Hey Xyla" using Porcupine (free tier, offline).
    Falls back to a keyboard trigger for development.
    """

    def __init__(self, on_wake: callable):
        self.on_wake = on_wake
        self._porcupine = None
        self._running = False
        self._setup()

    def _setup(self):
        try:
            import pvporcupine
            # Free Porcupine access key from picovoice.ai
            access_key = os.environ.get("PORCUPINE_KEY", "")
            if access_key:
                self._porcupine = pvporcupine.create(
                    access_key=access_key,
                    keywords=["hey google"],  # Use "hey google" as placeholder
                    # For production: custom "Hey Xyla" keyword file from Picovoice console
                )
                log.info("Porcupine wake word detector ready")
            else:
                log.info("No PORCUPINE_KEY set — wake word disabled. Say 'hey xyla' via keyboard: press SPACE")
        except ImportError:
            log.info("pvporcupine not installed — run: pip install pvporcupine")

    def start(self):
        self._running = True
        if self._porcupine:
            threading.Thread(target=self._porcupine_loop, daemon=True).start()
        else:
            threading.Thread(target=self._keyboard_loop, daemon=True).start()

    def _porcupine_loop(self):
        import pvporcupine
        import pyaudio
        import struct

        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=self._porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self._porcupine.frame_length
        )

        log.info("Wake word detection active — say 'Hey Xyla'")

        while self._running:
            pcm = audio_stream.read(self._porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * self._porcupine.frame_length, pcm)
            result = self._porcupine.process(pcm)
            if result >= 0:
                log.info("Wake word detected!")
                self.on_wake()

    def _keyboard_loop(self):
        """Dev mode: press SPACE to trigger wake word."""
        log.info("Keyboard mode: press ENTER to trigger wake")
        while self._running:
            input()
            self.on_wake()

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════
# MAIN XYLA CONTROLLER
# ═══════════════════════════════════════════════════════════

class XylaController:
    """
    Top-level controller. Orchestrates all subsystems.
    """

    def __init__(self, api_key: str, camera_index: int = 0, headless: bool = False):
        self.api_key = api_key
        self.headless = headless

        log.info("=== XYLA INITIALISING ===")

        # Subsystems
        self.vision = XylaVisionEngine(
            camera_index=camera_index,
            width=1280,
            height=720,
        )
        self.ai = XylaAI(api_key=api_key)
        self.voice = VoiceIO()

        # Callbacks
        self.vision.on_person_detected = self._on_person_detected
        self.vision.on_person_left = self._on_person_left

        # Wake word
        self.wake_detector = WakeWordDetector(on_wake=self._on_wake)

        # State
        self._is_active_conversation = False
        self._last_auto_analysis = 0
        self._auto_analysis_interval = 120  # seconds between automatic look checks

        log.info("=== XYLA READY ===")

    def start(self):
        """Start all subsystems."""
        self.vision.start()
        self.wake_detector.start()
        log.info("All systems online")

        # Startup greeting
        time.sleep(2)  # Let camera warm up
        self.voice.speak("Xyla is online. Step in front of your mirror when you're ready.")

        self._main_loop()

    def _main_loop(self):
        """Main event loop."""
        log.info("Main loop running. Press Ctrl+C to exit.")

        try:
            while True:
                # Periodic auto-analysis when someone is at mirror
                if (self.vision.state.person_present and
                    time.time() - self._last_auto_analysis > self._auto_analysis_interval):
                    self._do_auto_analysis()

                time.sleep(1)

        except KeyboardInterrupt:
            log.info("Shutting down...")
            self.shutdown()

    def _on_person_detected(self, face_data):
        """Called when someone steps in front of mirror."""
        log.info("Person at mirror — sending greeting")
        response = self.ai.wake_greeting()
        self.voice.speak(response.text)
        self._last_auto_analysis = time.time()

    def _on_person_left(self):
        """Called when person leaves mirror area."""
        log.info("Person left mirror")
        self.ai.clear_history()

    def _on_wake(self):
        """Called when wake word "Hey Xyla" is detected."""
        log.info("Wake word triggered")
        self.voice.speak("I'm here.")
        self._is_active_conversation = True

        # Listen for command
        text = self.voice.listen(duration=6.0)
        if text:
            self._handle_command(text)
        else:
            self.voice.speak("I didn't catch that — try again.")

        self._is_active_conversation = False

    def _handle_command(self, text: str):
        """Route a voice command to the right handler."""
        text_lower = text.lower()

        if any(w in text_lower for w in ["makeup", "face", "eyes", "lips", "blush", "contour"]):
            self._do_makeup_analysis()
        elif any(w in text_lower for w in ["outfit", "clothes", "wearing", "dress", "look"]):
            self._do_outfit_analysis()
        elif any(w in text_lower for w in ["affirmation", "inspire", "motivate", "confidence"]):
            self._do_affirmation()
        elif any(w in text_lower for w in ["photo", "picture", "capture", "selfie"]):
            self._do_capture()
        elif any(w in text_lower for w in ["full look", "everything", "whole look", "ready"]):
            self._do_full_analysis()
        else:
            # General chat
            context = self._build_context()
            response = self.ai.chat(text, context=context)
            self.voice.speak(response.text)

    def _do_makeup_analysis(self):
        """Analyse makeup and speak feedback."""
        log.info("Running makeup analysis")
        self.voice.speak("Let me check your makeup...")

        image_b64 = self.vision.capture_for_ai(region="face")
        if not image_b64:
            self.voice.speak("I can't see you clearly — step a little closer.")
            return

        face_context = self.vision.get_face_summary()
        response = self.ai.analyse_makeup(image_b64, face_context=face_context)
        self.voice.speak(response.text)
        log.info(f"Makeup analysis took {response.took_seconds}s")

    def _do_outfit_analysis(self):
        """Analyse outfit and speak feedback."""
        log.info("Running outfit analysis")
        self.voice.speak("Let me check your outfit...")

        image_b64 = self.vision.capture_for_ai(region="outfit")
        if not image_b64:
            self.voice.speak("Step back a little so I can see your full outfit.")
            return

        response = self.ai.analyse_outfit(image_b64)
        self.voice.speak(response.text)

    def _do_full_analysis(self):
        """Full look analysis."""
        log.info("Running full look analysis")
        self.voice.speak("Scanning your full look...")

        image_b64 = self.vision.capture_for_ai(region="full")
        if not image_b64:
            self.voice.speak("I can't get a clear view. Try stepping back a bit.")
            return

        response = self.ai.full_look_analysis(image_b64)
        self.voice.speak(response.text)

    def _do_affirmation(self):
        """Generate and speak an affirmation."""
        log.info("Generating affirmation")
        context = self._build_context()
        response = self.ai.get_affirmation(context=context)
        self.voice.speak(response.text)

    def _do_capture(self):
        """Take a photo."""
        log.info("Capturing photo")
        try:
            path = self.vision.save_capture("mirror_selfie")
            self.voice.speak(f"Photo saved! You can find it in the captures folder.")
            log.info(f"Photo saved: {path}")
        except Exception as e:
            self.voice.speak("Couldn't save the photo right now.")
            log.error(f"Capture failed: {e}")

    def _do_auto_analysis(self):
        """Periodic automatic check-in."""
        log.info("Running periodic auto-analysis")
        self._last_auto_analysis = time.time()

        image_b64 = self.vision.capture_for_ai(region="full")
        if not image_b64:
            return

        # Quick light check
        state = self.vision.get_state()
        if state.lighting_quality == "low":
            self.voice.speak("Quick tip — the lighting is quite dim right now. Move closer to a lamp for a more accurate view.")

    def _build_context(self) -> dict:
        """Build context dict for AI calls."""
        now = datetime.now()
        return {
            "current_time": now.strftime("%H:%M"),
            "face_detected": self.vision.state.face_detected,
            "lighting": self.vision.state.lighting_quality,
        }

    def shutdown(self):
        """Clean shutdown."""
        self.vision.stop()
        self.wake_detector.stop()
        self.voice.speak("Goodbye. You looked amazing today.")
        log.info("Xyla shutdown complete")


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Xyla AI Mirror")
    parser.add_argument("--api-key", type=str, default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--preview", action="store_true",
                        help="Show local preview window (desktop testing)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display (server mode)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key provided.")
        print("Set ANTHROPIC_API_KEY environment variable or use --api-key flag")
        sys.exit(1)

    if args.preview:
        # Desktop preview mode — just show the camera feed
        log.info("Running in preview mode (no AI, no voice)")
        engine = XylaVisionEngine(camera_index=args.camera)
        engine.run_preview()
    else:
        # Full Xyla mode
        controller = XylaController(
            api_key=args.api_key,
            camera_index=args.camera,
            headless=args.headless
        )
        controller.start()


if __name__ == "__main__":
    main()