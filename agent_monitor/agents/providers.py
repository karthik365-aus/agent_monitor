from __future__ import annotations

import os
import threading

from .base import Agent

# Limit concurrent Gemini requests to avoid hitting per-minute rate limits.
# 4 parallel workers all hitting Gemini simultaneously causes 429s; cap at 2.
_GEMINI_SEMAPHORE = threading.Semaphore(2)


class GroqAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from groq import Groq

        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def _call(self, prompt: str) -> tuple[str, dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        r = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        u = r.usage
        return r.choices[0].message.content, {
            "in": getattr(u, "prompt_tokens", 0),
            "out": getattr(u, "completion_tokens", 0),
        }


class GeminiAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=self.timeout),
        )

    def _call(self, prompt: str) -> tuple[str, dict]:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt or None,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        with _GEMINI_SEMAPHORE:
            r = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        um = getattr(r, "usage_metadata", None)
        usage = {
            "in": getattr(um, "prompt_token_count", 0),
            "out": getattr(um, "candidates_token_count", 0),
        }
        return (r.text or ""), usage
