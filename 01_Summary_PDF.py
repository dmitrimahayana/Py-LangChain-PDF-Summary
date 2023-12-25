import os

from langchain.document_loaders import PyPDFLoader

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma

# from langchain.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    folder_path = './Test Input/'
    filename = 'CV Dmitri Yanno Mahayana - Data - 2023-08 v1.9.pdf'
    loader = PyPDFLoader(folder_path + filename, extract_images=True)
    docs = loader.load_and_split()
    for page in docs:
        print(page.page_content)

    # os.environ['OPENAI_API_KEY'] = os.environ.get('CHATGPT_API_KEY')
    # faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    # docs = faiss_index.similarity_search("how many experience is dmitri?", k=2)
    # for doc in docs:
    #     print(str(doc.metadata["page"]) + ":", doc.page_content[:300])

    # model_folder_path = "C:/Users/dmitr/AppData/Local/nomic.ai/GPT4All/"
    # model_name = "llama-2-7b-chat.ggmlv3.q4_0.bin"
    # vectorstore = Chroma.from_documents(documents=pages,
    #                                     embedding=GPT4AllEmbeddings())
    # question = "who is dmitri yanno mahayana?"
    # docs = vectorstore.similarity_search(question)
    # print(len(docs))
    # print(docs[0])

    # loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    # data = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    # all_splits = text_splitter.split_documents(data)
    # for page in all_splits:
    #     print(page.page_content)
