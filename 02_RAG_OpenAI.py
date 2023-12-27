import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime

# Load PDF
print("\nStart Loading =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
folder_path = './Test Input/'
filename = 'journal_llama2.pdf'
loader = PyPDFLoader(folder_path + filename, extract_images=True)
docs = loader.load()
print("End Loading =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# Split text
print("\nStart Splitting-Storing-Retriever =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(docs)

# Store split results
os.environ['OPENAI_API_KEY'] = os.environ.get('CHATGPT_API_KEY')
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# retrieved_docs = retriever.get_relevant_documents(
#     "What is Truthfulness?"
# )
# print(len(retrieved_docs))
# print(retrieved_docs[0].page_content)
print("End Splitting-Storing-Retriever =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# RAG Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Adding Memory
condense_q_system_prompt = """Given a chat history and the latest user question \
which might reference the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
condense_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
condense_q_chain = condense_q_prompt | llm | StrOutputParser()

print("\nStart Condense Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
message = condense_q_chain.invoke(
    {
        "chat_history": [
            HumanMessage(content="What does LLM stand for?"),
            AIMessage(content="Large language model in machine learning world"),
        ],
        "question": "What does LLM mean?",
    }
)
print("End Condense Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def condense_question(input: dict):
    if input.get("chat_history"):
        return condense_q_chain
    else:
        return input["question"]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chaining
rag_chain = (
        RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
        | qa_prompt
        | llm
)
chat_history = []

print("\nStart Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
question = "What does LLM stand for?"
ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg])
print(f"{question}", ai_msg.content)
print("End Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

print("\nStart Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
question = "What is Bias?"
ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg])
print(f"{question}", ai_msg.content)
print("End Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

print("\nStart Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
question = "What are common example of that issues?"
ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg])
print(f"{question}", ai_msg.content)
print("End Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
