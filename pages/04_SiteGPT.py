from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models.openai import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

import streamlit as st

# classes

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs): # can receive as many arguments and keyword arguments
        self.message_box = st.empty() # empty widget you can later put things inside of

    def on_llm_end(self, *args, **kwargs): # can receive as many arguments and keyword arguments
        save_message(self.message, "ai")
    
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token # self. because inside of class
        self.message_box.markdown(self.message)

# prompts

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't, just say you don't know, do not make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question, the score should be high. If not, it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
    """
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Do not include any information about scores and date. Only include the link to the source.

            Answers: {answers}

            You may also use your past conversation and chat history to answer questions and have casual conversation.

            Chat History: {history}
            """,
        ),
        ("human", "{question}"),
    ]
)

# Chat History: {history} < add to implement chat history

# functions for llm

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
    )

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/(ai-gateway|vectorize|workers-ai)\/).*$"
        ],
        parsing_function=parse_page,
    )

    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
            "history": load_message()
        }
    )

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | temp
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


# functions for chat

def load_message():
    history = []

    for message in st.session_state["messages"]:
        tup = (f"{message['role']}", f"{message['message']}")
        history.append(tup)

    return ChatPromptTemplate.from_messages(history)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], False)

def save_message(message, role):
    st.session_state["messages"].append(
        {
            "message": message,
            "role": role
        }
    )

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)

    if save:
        save_message(message, role)

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🕸️"
)

st.markdown(
    """
    # Site GPT

    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
    """
)

with st.sidebar:
    st.write("This project is brought to you by:\nhttps://github.com/seansanghalee/fullstack_gpt")
    OPENAI_API_KEY = st.text_input("Your OpenAI API Key")
    url = st.text_input(
        "Enter URL",
        placeholder="https://example.com",
    )

if url and OPENAI_API_KEY:
    if ".xml" not in url:
        st.error("Please provide a sitemap ending in .xml")
    else:
        temp = ChatOpenAI(
                    temperature=0.0,
                    api_key=OPENAI_API_KEY if OPENAI_API_KEY else None,
                    streaming=True,
        )

        llm = ChatOpenAI(
            temperature=0.0,
            api_key=OPENAI_API_KEY if OPENAI_API_KEY else None,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
        )

        # get a retriever of the url
        retriever = load_website(url)

        # fire up chatbot
        send_message("I'm ready, ask me anything about the website!", "ai", False)
        paint_history()

        message = st.chat_input("Ask anything about the website...")
        if message:
            send_message(message, "human")

            # create chain
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            
            with st.chat_message("ai"):
                response = chain.invoke(message)

else:
    st.session_state["messages"] = []