import streamlit as st
from streamlit_chat import message
import pandas as pd
from llm import chat_with_data_api, info
from llm import extract_python_code,sidebar
import websockets
import asyncio
import base64
import json
import pyaudio

st.title("ChartEase")
st.markdown("Upload your files and ask the chatbot to generate graphs, ask questions")

api_key = ""
assembly_apikey = ""
max_tokens,api_key,assembly_apikey = info()

sidebar()

if 'text' not in st.session_state:
		st.session_state['text'] = 'Listening...'
		st.session_state['run'] = False

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()
 
# starts recording
stream = p.open(
   format=FORMAT,
   channels=CHANNELS,
   rate=RATE,
   input=True,
   frames_per_buffer=FRAMES_PER_BUFFER
)

URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

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
    
    if "context" not in st.session_state:
        st.session_state["context"] = [{"role": "system", "content": prompt}]
else:
    df = pd.DataFrame([])

# Define your custom avatar URLs (replace with actual URLs)
user_avatar_url = "https://api.dicebear.com/9.x/open-peeps/svg?seed=Luna"
assistant_avatar_url = "https://api.dicebear.com/9.x/shapes/svg?seed=Jasper"

def change():
     st.session_state['run'] = not st.session_state['run']

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.text_input("Enter your query",value=st.session_state.user_input.replace("."," "))
    st.session_state.user_input = ""
with col2:
    mic = st.button('🎙️', on_click=change)

async def send_receive():
    
    print(f'Connecting websocket to url ${URL}')

    async with websockets.connect(
        URL,
        extra_headers=(("Authorization", assembly_apikey),),
        ping_interval=5,
        ping_timeout=20
    ) as _ws:

        r = await asyncio.sleep(0.1)
        print("Receiving SessionBegins ...")

        session_begins = await _ws.recv()
        print(session_begins)
        print("Sending messages ...")


        async def send():
            while st.session_state['run']:
                try:
                    data = stream.read(FRAMES_PER_BUFFER)
                    data = base64.b64encode(data).decode("utf-8")
                    json_data = json.dumps({"audio_data":str(data)})
                    r = await _ws.send(json_data)

                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break

                except Exception as e:
                    print(e)
                    assert False, "Not a websocket 4008 error"

                r = await asyncio.sleep(0.01)


        async def receive():
            while st.session_state['run']:
                try:
                    result_str = await _ws.recv()
                    result = json.loads(result_str)['text']

                    if json.loads(result_str)['message_type']=='FinalTranscript':
                        st.session_state.user_input += result
                        print(result)
                        st.session_state['text'] = result
                        st.markdown(st.session_state['text'])

                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break

                except Exception as e:
                    print(e)
                    assert False, "Not a websocket 4008 error"
            
        send_result, receive_result = await asyncio.gather(send(), receive())


asyncio.run(send_receive())


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role":"assistant","content":"Please upload your data"})

model_params = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "max_tokens": max_tokens,
    "top_p": 0.5,
}

if mic and not assembly_apikey:
    st.warning("Enter your AssemblyAI key in the sidebar", icon="⚠")

if ("context" in st.session_state) and \
            (len(st.session_state["context"]) > 6):
        # Keep only the system prompt and the last `memory_window` prompts/answers
        st.session_state["context"] = (
            # the first one is always the system prompt
            [st.session_state["context"][0]]
            + st.session_state["context"][-(4):]
        )

if user_input:
    if api_key == "":
        st.warning("Enter your OpenAi key in the sidebar", icon="⚠")
    if df.empty:
        st.warning("Dataframe is empty, upload a valid file", icon="⚠")
    else:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["context"].append({"role": "user", "content": user_input})

        response = chat_with_data_api(df, api_key, **model_params)
        if response is not None:
            st.session_state["messages"].append(
                {"role": "assistant", "content": response})
            st.session_state["context"].append(
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
