from __future__ import annotations

from typing import Literal, TypedDict

# Retired models (see https://ai.google.dev/gemini-api/docs/changelog)
GEMINI_MODEL_ALIASES: dict[str, str] = {
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-1.5-flash-8b": "gemini-2.5-flash",
    "gemini-1.5-pro": "gemini-2.5-flash",
    "gemini-pro": "gemini-2.5-flash",
}

# Tried in order if the requested model is not found
GEMINI_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.0-flash",
)


class RestaurantResult(TypedDict):
    restaurant_name: str
    menu_items: str


def _build_prompt(cuisine: str) -> str:
    return (
        "You are a helpful assistant for restaurant branding.\n"
        f"Cuisine: {cuisine}\n\n"
        "Return EXACTLY in this format:\n"
        "Restaurant Name: <name>\n"
        "Menu Items: <comma separated items>\n"
    )


def _openai_generate(*, api_key: str, model: str, temperature: float, cuisine: str) -> RestaurantResult:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": _build_prompt(cuisine)}],
    )
    text = (resp.choices[0].message.content or "").strip()
    return _parse_result(text)


def _gemini_model_candidates(model: str) -> list[str]:
    requested = (model or "").strip()
    if requested in GEMINI_MODEL_ALIASES:
        requested = GEMINI_MODEL_ALIASES[requested]

    candidates: list[str] = []
    for name in (requested, *GEMINI_FALLBACK_MODELS):
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def _extract_gemini_text(resp) -> str:
    text = getattr(resp, "text", None)
    if text:
        return text.strip()

    candidates = getattr(resp, "candidates", None) or []
    if candidates:
        parts = getattr(candidates[0].content, "parts", None) or []
        chunks = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
        if chunks:
            return "\n".join(chunks).strip()

    return ""


def _gemini_generate(*, api_key: str, model: str, temperature: float, cuisine: str) -> RestaurantResult:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions

    genai.configure(api_key=api_key)
    prompt = _build_prompt(cuisine)
    last_error: Exception | None = None

    for model_name in _gemini_model_candidates(model):
        try:
            m = genai.GenerativeModel(model_name=model_name)
            resp = m.generate_content(
                prompt,
                generation_config={"temperature": temperature},
            )
            text = _extract_gemini_text(resp)
            if not text:
                raise RuntimeError("Gemini returned an empty response. Try again or pick another model.")
            return _parse_result(text)
        except google_exceptions.NotFound as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "No supported Gemini model was found. Use gemini-2.5-flash or gemini-3.5-flash "
        "(gemini-1.5-flash is retired)."
    ) from last_error


def _parse_result(text: str) -> RestaurantResult:
    name = ""
    items = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("restaurant name:"):
            name = line.split(":", 1)[1].strip()
        elif line.lower().startswith("menu items:"):
            items = line.split(":", 1)[1].strip()

    if not name and not items:
        return {"restaurant_name": "", "menu_items": text}

    return {"restaurant_name": name, "menu_items": items}


def generate_restaurant_name_and_items(
    cuisine: str,
    *,
    provider: Literal["openai", "gemini"],
    api_key: str,
    model: str,
    temperature: float = 0.7,
) -> RestaurantResult:
    provider_norm = (provider or "").strip().lower()
    if provider_norm == "openai":
        return _openai_generate(
            api_key=api_key,
            model=model,
            temperature=temperature,
            cuisine=cuisine,
        )
    if provider_norm == "gemini":
        return _gemini_generate(
            api_key=api_key,
            model=model,
            temperature=temperature,
            cuisine=cuisine,
        )
    raise ValueError("provider must be 'openai' or 'gemini'")
