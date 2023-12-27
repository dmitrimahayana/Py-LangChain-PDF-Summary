from langchain.document_loaders import PyPDFLoader

if __name__ == "__main__":
    folder_path = './Test Input/'
    filename = 'CV Dmitri Yanno Mahayana - Data - 2023-08 v1.9.pdf'
    loader = PyPDFLoader(folder_path + filename, extract_images=True)
    docs = loader.load_and_split()
    for page in docs:
        print(page.page_content)