# Restaurant Name Generator (Streamlit)

Generate a **restaurant name** and **menu items** using **OpenAI** or **Gemini**, and deploy on **Streamlit Community Cloud**.

## Run locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

## API keys

The app asks for your API key in the sidebar at runtime.

If deploying on Streamlit Community Cloud, add secrets:

- `OPENAI_API_KEY` (optional if you prefer using the sidebar)
- `GOOGLE_API_KEY` (optional if you prefer using the sidebar)

## Deploy (Streamlit Community Cloud)

- **Repo**: this GitHub repository
- **Main file path**: `main.py`
- **Python version**: 3.10+ recommended

