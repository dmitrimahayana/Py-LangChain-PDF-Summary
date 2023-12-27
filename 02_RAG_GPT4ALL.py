from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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
vectorstore = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings())

# Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# retrieved_docs = retriever.get_relevant_documents(
#     "What is Toxicity?"
# )
# print(len(retrieved_docs))
# print(retrieved_docs[0].page_content)
print("End Splitting-Storing-Retriever =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# LLM
model_folder_path = "C:/Users/dmitr/AppData/Local/nomic.ai/GPT4All/"
model_name = "llama-2-7b-chat.ggmlv3.q4_0.bin"
callbacks = [StreamingStdOutCallbackHandler()]
local_path = (
        model_folder_path + model_name
)
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

# Adding Memory
condense_system_prompt = """Given a chat history and the latest user question \
which might reference the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
condense_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
condense_chain = condense_prompt | llm | StrOutputParser()

print("\nStart Condense Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
message = condense_chain.invoke(
    {
        "chat_history": [
            HumanMessage(content="What does LLM stand for?"),
            AIMessage(content="Large language model in machine learning world"),
        ],
        "question": "What does LLM mean?",
    }
)
print("\nEnd Condense Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

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
        return condense_chain
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
chat_history.extend([HumanMessage(content=question), AIMessage(content=ai_msg)])
print("End Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

print("\nStart Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
question = "What is Toxicity?"
ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), AIMessage(content=ai_msg)])
print("\nEnd Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

print("\nStart Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
question = "What is Bias?"
ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), AIMessage(content=ai_msg)])
print("\nEnd Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

print("\nStart Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
question = "What are common example of that issues?"
ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), AIMessage(content=ai_msg)])
print("\nEnd Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
