import re
import openai
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.schema.output_parser import OutputParserException

def info():
    with st.sidebar:
        max_tokens = st.slider(
                label="Maximum length (tokens)",
                value=256,
                min_value=0,
                max_value=4096,
                step=1,
                help=(
                    """The maximum number of tokens to generate in the chat completion.
                    The total length of input tokens and generated tokens is limited by
                    the model's context length."""
                )
            )
        api_key = st.text_input(
            label="Enter your OpenAI key",
            type="password",
            help=(
                """Obtain your key from the OpenAI website, 
                then paste it into the "Enter Your OpenAI Key" section on our site and click "Save" or "Submit."
                If you encounter any issues, please contact our support team for assistance."""
            )
        )
    return max_tokens, api_key

def extract_python_code(text):
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    else:
        return matches[0]

def plot_matplotlib_code(code):
    # Redirect the output to a BytesIO object
    buf = BytesIO()
    exec(code)
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f"data:image/png;base64,{image}"

def chat_with_data_api(df, api_key, model="gpt-3.5-turbo", temperature=0.0, max_tokens=256, top_p=0.5):
    """
    A function that answers data questions from a dataframe.
    """
    if "plot" in st.session_state.messages[-1]["content"].lower():
        code_prompt = """
            Generate the code <code> for plotting the previous data in plotly,
            in the format requested. The solution should be given using plotly
            and only plotly. Do not use matplotlib.
            Return the code <code> in the following
            format ```python <code>```
        """
        st.session_state.messages.append({
            "role": "assistant",
            "content": code_prompt
        })
        response = openai.ChatCompletion.create(
            model=model,
            messages=st.session_state.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            api_key=api_key
        )
        code = extract_python_code(response["choices"][0]["message"]["content"])
        if code is None:
            st.warning(
                "Couldn't find data to plot in the chat. "
                "Check if the number of tokens is too low for the data at hand. "
                "I.e. if the generated code is cut off, this might be the case.",
                icon="🚨"
            )
            return "Couldn't plot the data"
        else:
            code = code.replace("fig.show()", "")
            code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""  # noqa: E501
            st.write(f"```{code}")
            exec(code)
            return response["choices"][0]["message"]["content"]
    else:
        
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            openai_api_key=api_key
        )

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            return_intermediate_steps=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=False,
            allow_dangerous_code=True
        )

        try:
            answer = pandas_df_agent(st.session_state.messages)
            if answer["intermediate_steps"]:
                action = answer["intermediate_steps"][-1][0].tool_input["query"]
                st.write(f"Executed the code ```{action}```")
                
                if "matplotlib" in action:
                    # Generate the plot and return the image
                    image_data = plot_matplotlib_code(action)
                    st.image(image_data)
                    return "Matplotlib plot generated."
                    
            return answer["output"]
        except OutputParserException:
            error_msg = """OutputParserException error occured in LangChain agent.
                Refine your query."""
            return error_msg
        except:  # noqa: E722
            answer = "Unknown error occured in LangChain agent. Refine your query"
            return answer


