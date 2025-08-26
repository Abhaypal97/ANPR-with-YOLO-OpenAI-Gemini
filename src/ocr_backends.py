import base64
from typing import Optional


class DummyOCR:
    def extract_text(self, image_b64: str) -> Optional[str]:
        return "PLATE-XXXX"


class OpenAIOCR:
    def __init__(self, api_key: str) -> None:
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(api_key=api_key)

    def extract_text(self, image_b64: str) -> Optional[str]:
        prompt = (
            "Extract license plate text. If not sure, return None. Only output text."
        )
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }
            ],
        )
        return (resp.choices[0].message.content or "").strip() or None


class GeminiOCR:
    def __init__(self, api_key: str) -> None:
        from google import genai  # type: ignore
        from google.genai import types  # noqa: F401  # type: ignore

        self.client = genai.Client(api_key=api_key)

    def extract_text(self, image_b64: str) -> Optional[str]:
        from google.genai import types  # type: ignore

        image = types.Blob(mime_type="image/jpeg", data=base64.b64decode(image_b64))
        prompt = "Extract license plate text from the image. Only output text or None."
        resp = self.client.models.generate_content(
            model="gemini-1.5-flash-8b",
            contents=[prompt, image],
            config=types.GenerateContentConfig(temperature=0.2),
        )
        text = (resp.text or "").strip()
        return text or None


