from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime

# # Load document
# print("\nStart Scraping =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# docs = loader.load()
# print("End Scraping =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# Load and Split
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

# RAG Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
model_folder_path = "C:/Users/dmitr/AppData/Local/nomic.ai/GPT4All/"
model_name = "llama-2-7b-chat.ggmlv3.q4_0.bin"
callbacks = [StreamingStdOutCallbackHandler()]
local_path = (
        model_folder_path + model_name
)
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)


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
question = "What is Toxicity?"
result = rag_chain.invoke(question)
print(f"\n{question}: ", result.strip())
print("End Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))