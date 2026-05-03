"""
XYLA AI Analysis Module
=======================
Connects the vision engine to Claude API.
Handles makeup analysis, outfit feedback, affirmations, and conversation.

Author: Xyla Project
"""

import anthropic
import base64
import json
import time
import threading
import logging
from typing import Optional, Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════

@dataclass
class XylaResponse:
    text: str
    analysis_type: str      # "makeup" | "outfit" | "chat" | "affirmation"
    took_seconds: float
    had_image: bool


XYLA_SYSTEM_PROMPT = """
You are Xyla — a warm, empowering, and brilliantly observant AI mirror assistant. 
You exist to help women feel confident, beautiful, and fully prepared to show up in the world.

Your personality:
- Warm and affirming, never harsh or critical
- Specific and actionable — you notice real details, not generalities
- Occasionally playful, always genuine
- You celebrate effort and beauty, while offering one or two useful tips
- You speak like a supportive best friend who happens to be a professional stylist

Your capabilities:
- You can SEE the person through the mirror camera
- You analyse makeup application, blending, symmetry, and technique
- You review outfit proportions, colour harmony, and styling
- You give personalised daily affirmations
- You can see their calendar and schedule
- You are aware of the time of day and can tailor advice accordingly

Rules:
- Never make someone feel bad about their appearance
- Always find something genuine and specific to compliment first
- Keep responses to 2-5 sentences unless asked for more
- If you see a real issue (e.g. mascara smudge, lipstick on teeth), mention it kindly and specifically — this is genuinely helpful
- Use occasional emojis — but not excessively
- Speak in second person ("you look amazing") not third
""".strip()


# ═══════════════════════════════════════════════════════════
# AI ENGINE
# ═══════════════════════════════════════════════════════════

class XylaAI:
    """
    Handles all AI interactions for Xyla.
    Connects to Claude API with vision support.

    Usage:
        ai = XylaAI(api_key="sk-ant-...")
        
        # Text chat
        response = ai.chat("How do I do a smoky eye?")
        
        # Vision analysis
        response = ai.analyse_makeup(image_b64)
        response = ai.analyse_outfit(image_b64)
        
        # Affirmation
        response = ai.get_affirmation()
    """

    MODEL = "claude-opus-4-5"
    MAX_TOKENS = 400

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history = []
        self._lock = threading.Lock()
        log.info("Xyla AI engine initialised")

    def _call(
        self,
        user_content,
        system_extra: str = "",
        max_tokens: int = None
    ) -> str:
        """Make a Claude API call and return the text response."""
        system = XYLA_SYSTEM_PROMPT
        if system_extra:
            system += f"\n\n{system_extra}"

        with self._lock:
            self.conversation_history.append({
                "role": "user",
                "content": user_content
            })

            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=max_tokens or self.MAX_TOKENS,
                system=system,
                messages=self.conversation_history
            )

            reply = response.content[0].text

            self.conversation_history.append({
                "role": "assistant",
                "content": reply
            })

            # Trim history to last 20 turns to avoid context overflow
            if len(self.conversation_history) > 40:
                self.conversation_history = self.conversation_history[-40:]

            return reply

    def _build_image_content(self, image_b64: str, text: str) -> list:
        """Build multimodal message content with image + text."""
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64
                }
            },
            {
                "type": "text",
                "text": text
            }
        ]

    # ──────────────────────────────────────────────────────
    # CORE ANALYSIS METHODS
    # ──────────────────────────────────────────────────────

    def analyse_makeup(self, image_b64: str, face_context: dict = None) -> XylaResponse:
        """
        Analyse makeup in a mirror frame.

        Args:
            image_b64: Base64 JPEG of the face/mirror
            face_context: Optional dict from vision_engine.get_face_summary()

        Returns:
            XylaResponse with makeup feedback
        """
        start = time.time()

        context_str = ""
        if face_context:
            if face_context.get("lighting_quality") == "low":
                context_str = "Note: lighting is low which may affect colour accuracy. "
            elif face_context.get("lighting_quality") == "harsh":
                context_str = "Note: harsh lighting detected — shadows may be pronounced. "
            if face_context.get("eye_symmetry_score"):
                sym = face_context["eye_symmetry_score"]
                if sym < 0.75:
                    context_str += f"Eye asymmetry detected (score: {sym:.2f}). "

        prompt = f"""Please analyse my makeup carefully. {context_str}

Look specifically for:
1. Blending — are there any harsh lines or unblended edges?
2. Symmetry — are both sides balanced?
3. Coverage — any missed spots, patches, or uneven foundation?
4. Eye makeup — liner, shadow, mascara — any smudges or issues?
5. Lips — is the lip line clean? Any colour outside the lips?

Start with what looks great, then give me 1-2 specific things to fix if anything needs it."""

        content = self._build_image_content(image_b64, prompt)
        reply = self._call(content)

        return XylaResponse(
            text=reply,
            analysis_type="makeup",
            took_seconds=round(time.time() - start, 2),
            had_image=True
        )

    def analyse_outfit(self, image_b64: str, occasion: str = None) -> XylaResponse:
        """
        Analyse outfit in a mirror frame.

        Args:
            image_b64: Base64 JPEG of full/partial body
            occasion: Optional context like "work", "date", "casual"

        Returns:
            XylaResponse with outfit feedback
        """
        start = time.time()

        occasion_str = f" I'm dressing for {occasion}." if occasion else ""

        prompt = f"""Please review my outfit.{occasion_str}

Look at:
1. Proportions — do the top and bottom balance well?
2. Colour harmony — do the pieces work together?
3. Fit — does it look like the right size/shape for my body?
4. Styling details — any finishing touches that would elevate the look?
5. Overall vibe — what energy does this outfit give?

Tell me what you love about it first, then one specific tip to make it even better."""

        content = self._build_image_content(image_b64, prompt)
        reply = self._call(content)

        return XylaResponse(
            text=reply,
            analysis_type="outfit",
            took_seconds=round(time.time() - start, 2),
            had_image=True
        )

    def full_look_analysis(self, image_b64: str) -> XylaResponse:
        """
        Complete head-to-toe look analysis.
        Combines makeup + outfit + overall energy in one call.
        """
        start = time.time()

        prompt = """Give me a complete look analysis — makeup, outfit, and overall energy.

Be specific about:
- What's working really well (be precise, not generic)
- One makeup tip if anything could be improved
- One outfit/styling tip if relevant
- The overall impression you get — what does this look say?

Keep it under 6 sentences total. Make me feel seen and confident."""

        content = self._build_image_content(image_b64, prompt)
        reply = self._call(content)

        return XylaResponse(
            text=reply,
            analysis_type="full_look",
            took_seconds=round(time.time() - start, 2),
            had_image=True
        )

    def chat(self, message: str, context: dict = None) -> XylaResponse:
        """
        General conversation with Xyla.

        Args:
            message: User's message
            context: Optional mirror state context

        Returns:
            XylaResponse with conversational reply
        """
        start = time.time()

        system_extra = ""
        if context:
            if context.get("current_time"):
                system_extra += f"Current time: {context['current_time']}. "
            if context.get("events_today"):
                system_extra += f"User's schedule today: {', '.join(context['events_today'])}. "
            if context.get("face_detected"):
                system_extra += "The user is currently standing in front of the mirror. "

        reply = self._call(message, system_extra=system_extra, max_tokens=300)

        return XylaResponse(
            text=reply,
            analysis_type="chat",
            took_seconds=round(time.time() - start, 2),
            had_image=False
        )

    def get_affirmation(self, context: dict = None) -> XylaResponse:
        """
        Generate a personalised affirmation.

        Args:
            context: Optional dict with time, events, etc.

        Returns:
            XylaResponse with affirmation
        """
        start = time.time()

        time_str = ""
        events_str = ""
        if context:
            if context.get("current_time"):
                hour = int(context["current_time"].split(":")[0])
                if hour < 10:
                    time_str = "It's early morning — they're starting their day."
                elif hour < 12:
                    time_str = "It's mid-morning."
                elif hour < 17:
                    time_str = "It's afternoon."
                else:
                    time_str = "It's evening — they may be getting ready to go out."
            if context.get("events_today"):
                events_str = f"They have these events today: {', '.join(context['events_today'][:3])}."

        prompt = f"""Give me one powerful, personalised affirmation to carry through my day. {time_str} {events_str}

Make it:
- Specific and vivid, not generic
- About inner power and showing up fully
- 1-2 sentences maximum
- Something that would make someone stop and feel it

Just the affirmation — no intro, no "here's your affirmation:" preamble."""

        reply = self._call(prompt, max_tokens=100)

        return XylaResponse(
            text=reply,
            analysis_type="affirmation",
            took_seconds=round(time.time() - start, 2),
            had_image=False
        )

    def wake_greeting(self, face_context: dict = None) -> XylaResponse:
        """
        Greeting when person first appears at mirror.
        Called by vision engine on_person_detected callback.
        """
        start = time.time()

        from datetime import datetime
        hour = datetime.now().hour
        if hour < 12:
            greeting_context = "Good morning"
        elif hour < 17:
            greeting_context = "Good afternoon"
        else:
            greeting_context = "Good evening"

        prompt = f"""{greeting_context}! Someone just stepped in front of the Xyla mirror. 
Give them a warm, brief welcome that makes them feel immediately seen and good about themselves. 
1-2 sentences. Be genuine, not robotic."""

        reply = self._call(prompt, max_tokens=100)

        return XylaResponse(
            text=reply,
            analysis_type="greeting",
            took_seconds=round(time.time() - start, 2),
            had_image=False
        )

    def clear_history(self):
        """Clear conversation history (new session)."""
        with self._lock:
            self.conversation_history = []
        log.info("Conversation history cleared")