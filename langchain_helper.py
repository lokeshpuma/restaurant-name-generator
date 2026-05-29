from __future__ import annotations

from typing import Literal, TypedDict


class RestaurantResult(TypedDict):
    restaurant_name: str
    menu_items: str


def _openai_generate(*, api_key: str, model: str, temperature: float, cuisine: str) -> RestaurantResult:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a helpful assistant for restaurant branding.\n"
        f"Cuisine: {cuisine}\n\n"
        "Return EXACTLY in this format:\n"
        "Restaurant Name: <name>\n"
        "Menu Items: <comma separated items>\n"
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (resp.choices[0].message.content or "").strip()
    return _parse_result(text)


def _gemini_generate(*, api_key: str, model: str, temperature: float, cuisine: str) -> RestaurantResult:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model_name=model)
    prompt = (
        "You are a helpful assistant for restaurant branding.\n"
        f"Cuisine: {cuisine}\n\n"
        "Return EXACTLY in this format:\n"
        "Restaurant Name: <name>\n"
        "Menu Items: <comma separated items>\n"
    )
    resp = m.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
        },
    )
    text = (getattr(resp, "text", "") or "").strip()
    return _parse_result(text)


def _parse_result(text: str) -> RestaurantResult:
    name = ""
    items = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("restaurant name:"):
            name = line.split(":", 1)[1].strip()
        elif line.lower().startswith("menu items:"):
            items = line.split(":", 1)[1].strip()

    # fallback: if model didn't respect format, just return whole text as menu
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
