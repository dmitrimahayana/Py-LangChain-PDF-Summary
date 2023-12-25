import os
import bs4
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime

# # Load document
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

# Load and Split
print("\nStart Loading =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
folder_path = './Test Input/'
filename = 'journal_llama2.pdf'
loader = PyPDFLoader(folder_path + filename, extract_images=True)
docs = loader.load()
print("End Loading =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# Split text
print("\nStart Splitting-Storing-Retriever =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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


# Chaining
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

print("\nStart Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
result = rag_chain.invoke("What is Toxicity?")
print(result)
print("End Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
