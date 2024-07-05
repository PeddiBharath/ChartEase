import streamlit as st
from streamlit_chat import message
import pandas as pd
from llm import chat_with_data_api, info
from llm import extract_python_code
st.title("ChartEase")
st.markdown("Upload your files and ask the chatbot to generate graphs, ask questions")

api_key = ""
assembly_apikey = ""
max_tokens, api_key,assembly_apikey = info()

uploaded_file = st.file_uploader(label="Choose file", type=["csv","xlsx","xls"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_extension in ['xlsx', 'xls']:
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format")
    
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

# Define your custom avatar URLs (replace with actual URLs)
user_avatar_url = "https://api.dicebear.com/9.x/open-peeps/svg?seed=Luna"
assistant_avatar_url = "https://api.dicebear.com/9.x/shapes/svg?seed=Jasper"


col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.chat_input("Enter your query")
with col2:
    mic = st.button('🎙️')

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role":"assistant","content":"Please upload your data"})

model_params = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "max_tokens": max_tokens,
    "top_p": 0.5,
}

if mic and assembly_apikey == "":
        st.warning("Enter your AssemblyAi key in the side bar", icon="⚠")

if user_input:
    if api_key == "":
        st.warning("Enter your OpenAi key in the sidebar", icon="⚠")
    if df.empty:
        st.warning("Dataframe is empty, upload a valid file", icon="⚠")
    else:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        response = chat_with_data_api(df, api_key, **model_params)
        if response is not None:
            st.session_state["messages"].append(
                {"role": "assistant", "content": response})

for message in st.session_state.messages:
    if message["role"] == "user":
        avatar_url = user_avatar_url
        with st.chat_message(message["role"], avatar=avatar_url):
            st.markdown(message["content"])
    else:
        avatar_url = assistant_avatar_url
        with st.chat_message(message["role"], avatar=avatar_url):
            if "import plotly" in message["content"]:
                code = extract_python_code(message["content"])
                code = code.replace("fig.show()", "")
                code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""
                exec(code)
            st.markdown(message["content"])
