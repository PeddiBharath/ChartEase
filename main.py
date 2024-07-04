import streamlit as st
from streamlit_chat import message
import pandas as pd
from llm import chat_with_data_api, info

st.title("Chat with, query and plot your own data")

max_tokens, api_key = info()

uploaded_file = st.file_uploader(label="Choose file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    prompt = f"""You are a python expert. You will be given questions for
        manipulating an input dataframe.
        The available columns are: {df.columns}.
        Use them for extracting the relevant data.
        IMPORTANT: Only use Plotly for plotting. Do not use Matplotlib.
    """
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": prompt}]
else:
    df = pd.DataFrame([])

# Storing the chat
if "generated" not in st.session_state:
    st.session_state["generated"] = ["Please upload your data"]

if "past" not in st.session_state:
    st.session_state["past"] = []

user_input = st.chat_input("Enter your query")
if ((len(st.session_state["past"]) > 0)
        and (user_input == st.session_state["past"][-1])):
    user_input = ""

if ("messages" in st.session_state) and \
        (len(st.session_state["messages"]) > 2 * 3):
    # Keep only the system prompt and the last memory_window prompts/answers
    st.session_state["messages"] = (
        # the first one is always the system prompt
        [st.session_state["messages"][0]]
        + st.session_state["messages"][-(2 * 3 - 2):]
    )

model_params = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "max_tokens": max_tokens,
    "top_p": 0.5,
}

if user_input:
    if df.empty:
        st.warning("Dataframe is empty, upload a valid file", icon="⚠")
    else:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        response = chat_with_data_api(df, api_key, **model_params)
        st.session_state.past.append(user_input)
        if response is not None:
            st.session_state.generated.append(response)
            st.session_state["messages"].append(
                {"role": "assistant", "content": response})

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        if i - 1 >= 0:
            message(
                st.session_state["past"][i - 1],
                is_user=True,
                key=str(i) + "_user"
            )
