import streamlit as st

st.set_page_config(
    page_title="FullstackGPT",
    page_icon="🤖",
)

st.title("Home")

st.markdown(
    """
    # Welcome!
                
    Here are the apps:
                
    - 📃 [Document GPT](/document)
    - ❓ [Quiz GPT](/quiz)
    -  [Video GPT](/video)
    - 💼 [Investor GPT](/investor)
    """
)