import streamlit as st
import langchain_helper

st.set_page_config(page_title="Restaurant Name Generator", page_icon="🍽️", layout="centered")
st.title("Restaurant Name Generator")
st.caption("Generate a restaurant name and menu items using OpenAI or Gemini.")

with st.sidebar:
    st.subheader("Model settings")
    provider = st.selectbox("Provider", ("OpenAI", "Gemini"), index=0)

    api_key_label = "OpenAI API key" if provider == "OpenAI" else "Gemini API key"
    api_key_help = (
        "Used only for this session in your browser."
        if provider == "OpenAI"
        else "This is your Google AI Studio (Gemini) API key. Used only for this session."
    )
    api_key = st.text_input(api_key_label, type="password", help=api_key_help)

    if provider == "OpenAI":
        model = st.text_input("Model", value="gpt-4o-mini")
    else:
        model = st.text_input("Model", value="gemini-1.5-flash")

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

st.divider()
cuisine = st.selectbox("Pick a cuisine", ("Indian", "Italian", "Mexican", "Arabic", "American"))

generate = st.button("Generate", type="primary", use_container_width=True)

if generate:
    if not api_key.strip():
        st.error("Please enter your API key in the sidebar.")
        st.stop()

    with st.spinner("Generating..."):
        response = langchain_helper.generate_restaurant_name_and_items(
            cuisine,
            provider="openai" if provider == "OpenAI" else "gemini",
            api_key=api_key.strip(),
            model=model.strip(),
            temperature=float(temperature),
        )

    restaurant_name = (response.get("restaurant_name") or "").strip()
    menu_raw = (response.get("menu_items") or "").strip()

    if restaurant_name:
        st.subheader(restaurant_name)
    else:
        st.subheader("Your restaurant")

    st.write("**Menu items**")
    # Support both comma-separated and bullet-style responses.
    if "," in menu_raw:
        items = [x.strip(" -\n\t") for x in menu_raw.split(",") if x.strip()]
    else:
        items = [x.strip(" -\n\t") for x in menu_raw.splitlines() if x.strip()]

    if not items:
        st.write(menu_raw or "(No menu items returned)")
    else:
        for item in items:
            st.write("-", item)

