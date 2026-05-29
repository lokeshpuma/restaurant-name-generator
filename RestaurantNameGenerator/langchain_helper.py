from __future__ import annotations

from dataclasses import dataclass

from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore


@dataclass(frozen=True)
class GeneratorConfig:
    provider: str  # "openai" | "gemini"
    api_key: str
    model: str
    temperature: float = 0.7


def _build_llm(cfg: GeneratorConfig):
    provider = (cfg.provider or "").strip().lower()
    if provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("Missing dependency: langchain-openai")
        return ChatOpenAI(model=cfg.model, temperature=cfg.temperature, api_key=cfg.api_key)

    if provider in {"gemini", "google"}:
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("Missing dependency: langchain-google-genai")
        return ChatGoogleGenerativeAI(
            model=cfg.model,
            temperature=cfg.temperature,
            google_api_key=cfg.api_key,
        )

    raise ValueError("provider must be 'openai' or 'gemini'")


def generate_restaurant_name_and_items(
    cuisine: str,
    *,
    provider: str,
    api_key: str,
    model: str,
    temperature: float = 0.7,
):
    llm = _build_llm(
        GeneratorConfig(provider=provider, api_key=api_key, model=model, temperature=temperature)
    )

    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this.",
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    prompt_template_items = PromptTemplate(
        input_variables=["restaurant_name"],
        template="Suggest some menu items for {restaurant_name}. Return it as a comma separated string.",
    )
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=["cuisine"],
        output_variables=["restaurant_name", "menu_items"],
    )
    return chain({"cuisine": cuisine})


if __name__ == "__main__":  # manual smoke test
    raise SystemExit(
        "Run the Streamlit app instead:\n\n  streamlit run RestaurantNameGenerator/main.py\n"
    )
