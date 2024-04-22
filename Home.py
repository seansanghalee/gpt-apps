import streamlit as st

# from langchain.prompts import PromptTemplate

# intro

# st.title("Hello world!")

# st.subheader("Welcome to Streamlit!")

# st.markdown("""
#     #### I love it
# """)

# st.write() -> streamlit magic

# st.write("hello")
# st.write([1, 2, 3, 4])
# st.write({"x": 1})

# st.write(PromptTemplate)

# p = PromptTemplate.from_template("xxxx")
# st.write(p)
# p

# model = st.selectbox("Choose your model", ("GPT-3", "GPT-4"))

# from datetime import datetime

# today = datetime.today().strftime("%H:%M:%S")
# st.write(today)
# st.write(model)

# value = st.slider("temperature", 0.1, 1.0)

# sidebar

st.title("SangderellaGPT")
# st.sidebar.title("Sidebar Title")
# st.sidebar.text_input("xxx")

st.markdown(
    """
    Welcome to my GPT portfolio :)
    
    Here are the apps I made:
    - [DocumentGPT](/DocumentGPT)
    - [PrivateGPT](/PrivateGPT)
    - [QuizGPT](/QuizGPT)
    - [SiteGPT](/SiteGPT)
    - MeetingGPT
    - [InvestorGPT](/InvestorGPT)
    - [AssistantGPT](/AssistantGPT)
    """
)

# same as

# with st.sidebar:
#     st.title("Sidebar Title")
#     st.text_input("xxx")
