import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
import json

class JsonOutputParser(BaseOutputParser):

    def parse(self, text):
        import json
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


st.set_page_config(
    page_title="QuizGPT",
    page_icon="🧐",
)

st.title("Quiz GPT")

with st.sidebar:
    st.write("This project is brought to you by:\nhttps://github.com/seansanghalee/fullstack_gpt")
    OPENAI_API_KEY = st.text_input("Your OpenAI API Key")
    difficulty = st.selectbox("Choose difficulty", (
        "Easy",
        "Medium",
        "Hard",
        "Impossible"
        )
    )

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string"
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            }
                        }
                    },
                    "required": ["question", "answers"],
                }
            }
        }
    }
}

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ],
    api_key=OPENAI_API_KEY
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function
    ]
)

output_parser = JsonOutputParser()

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    # create a copy of the file to work with
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    # os.makedirs(f"./.cache/quiz_files/", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    # load and split the file
    loader = UnstructuredFileLoader(file_path)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = '\n',
        chunk_size = 600,
        chunk_overlap = 100,
    )
    
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

# @st.cache_data(show_spinner="Generating quiz...")
# def run_quiz_chain(_docs, topic):
#     chain = {"context": questions_chain} | formatting_chain | output_parser
#     return chain.invoke(_docs)

@st.cache_data(show_spinner="Searching Wikipedia...")
def search_wiki(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(topic)

    return docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_data(show_spinner="Creating quiz...")
def invoke_chain(_chain, _docs, difficulty):
    response = _chain.invoke({"content": _docs, "difficulty": difficulty})
    return response


# questions_prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          """
#     You are a helpful assistant that is role playing as a teacher.

#     Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

#     Try to make the overall difficulty of the quiz {difficuly}.

#     Each question should have 4 answers, three of them must be incorrect and one should be correct.

#     Use (o) to signal the correct answer.

#     Question exaples:

#     Question: What is the color of the ocean?
#     Answers: Red | Yellow | Green | Blue(o)

#     Question: What is the capital of Georgia?
#     Answers: Baku | Tbilisi(o) | Manila | Beirut

#     Question: When was Avatar released?
#     Answers: 2007 | 2001 | 2009(o) | 1998

#     Question: Who was Julius Caesar?
#     Answers: A Roman Emperor(o) | Painter | Actor | Model

#     Your turn!

#     Context: {context}
#     """),
#  ])

# questions_chain = {
#     "context": format_docs,
#     "difficulty": difficulty,
# } | questions_prompt | llm

# formatting_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      """
#         You are a powerful formatting algorithm.

#         You format exam questions into JSON format.
#         Answers with (o) are the correct ones.

#         Example Input:

#         Question: What is the color of the ocean?
#         Answers: Red | Yellow | Green | Blue(o)

#         Question: What is the capital of Georgia?
#         Answers: Baku | Tbilisi(o) | Manila | Beirut

#         Question: When was Avatar released?
#         Answers: 2007 | 2001 | 2009(o) | 1998

#         Question: Who was Julius Caesar?
#         Answers: A Roman Emperor(o) | Painter | Actor | Model
        
#         Example Output:

#         ```json
#         {{ "questions": [
#             {{
#                 "question": "What is the color of the ocean?",
#                 "answers": [
#                     {{
#                         "answer": "Red",
#                         "correct": false
#                     }},
#                     {{
#                         "answer": "Yellow",
#                         "correct": false
#                     }},
#                     {{
#                         "answer": "Green",
#                         "correct": false
#                     }},
#                     {{
#                         "answer": "Blue",
#                         "correct": true
#                     }},
#                 ]
#             }},
#             {{
#                 "question": "What is the capital of Georgia?",
#                 "answers": [
#                     {{
#                         "answer": "Baku",
#                         "correct": false
#                     }},
#                     {{
#                         "answer": "Tbilisi",
#                         "correct": true
#                     }},
#                     {{
#                         "answer": "Manila",
#                         "correct": false
#                     }},
#                     {{
#                         "answer": "Beirut",
#                         "correct": false
#                     }},
#                 ]
#             }},
#             {{
#                 "question": "When was Avatar released?",
#                 "answers": [
#                     {{
#                         "answer": "2007",
#                         "correct": false
#                     }},
#                     {{
#                         "answer": "2001",
#                         "correct": false
#                     }},
#                     {{
#                         "answer": "2009",
#                         "correct": true
#                     }},
#                     {{
#                         "answer": "1998",
#                         "correct": false
#                     }},
#                 ]
#             }},
#             {{
#                 "question": "Who was Julius Caesar?",
#                 "answers": [
#                     {{
#                         "answer": "A Roman Emperor",
#                         "correct": true
#                     }},
#                     {{
#                         "answer": "Painter",
#                         "correct": false
#                     }},
#                     {{
#                         "answer": "Actor",
#                         "correct": false
#                     }},
#                     {{
#                         "answer": "Model",
#                         "correct": false
#                     }},
#                 ]
#             }},
#             ]
#         }}
#         ```

#     Your turn!

#     Context: {context}
#     """
#     ),
# ])

# formatting_chain = formatting_prompt | llm

with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox("Choose what you want to use.", (
        "File",
        "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader("Upload a .docx, .pdf, or .txt file", type=["docx", "pdf", "txt"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia")
        if topic:
            docs = search_wiki(topic)
            
if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
    
    I will make a quiz from Wikipedia articles or files you upload
    to test your knowledge and help you study.
    
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
# else:
#     response = run_quiz_chain(docs, topic if topic else file.name)
#     with st.form("questions_form"):
#         for question in response["questions"]:
#             st.write(question["question"])
#             value = st.radio("Select an option.", [answer["answer"] for answer in question["answers"]], index=None)
#             if {"answer": value, "correct": True} in question["answers"]:
#                 st.success("Correct!")
#             elif value:
#                 st.error("Wrong!")

#         button = st.form_submit_button()
#     st.balloons()
else: # using function calling
    quiz_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
    You are a helpful assistant that is role playing as a teacher.

    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    Try to make the overall difficulty of the quiz {difficulty}.

    Content: {content}
    """
        )
    ])
    chain = quiz_prompt| llm
    response = invoke_chain(chain, docs, difficulty)
    response = response.additional_kwargs["function_call"]["arguments"]
    response = json.loads(response)

    score = 0

    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio("Select an option.", [answer["answer"] for answer in question["answers"]], index=None)
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                score += 1
            elif value:
                st.error("Wrong!")

        submitted = st.form_submit_button()
    
    if submitted:
        if score == 10:
            st.write("Congratulations! You got all questions right 💯")
            st.balloons()
        else:
            st.write(f"You got {score} question(s) right!")
            retry = st.button("Retry?")