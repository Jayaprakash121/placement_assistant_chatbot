import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "database", "Student_Handbook_for_Internship_&_Placement-SHIP.pdf")
csv_path = os.path.join(current_dir, "database", "MASTERSHEET_SPC_2024-25-Placements.csv")
persistent_directory = os.path.join(current_dir, "db", "all_chroma_db")

def create_or_load_chroma_db():
    # Create embeddings
    print("Creating embeddings")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    print("Finished creating embeddings")

    # Check if the Chroma vector store already exists
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")

        # Ensure the text file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

        # Load Placement CSV File
        df = pd.read_csv(csv_path)
        csv_docs = []
        for _, row in df.iterrows():
            content = f"{row['Student Name']} (Roll: {row['Roll number']}) from {row['Branch']} got placed in {row['Company']} as {row['Role']} with a CTC of {row['CTC']} LPA."
            metadata = {
                "source": csv_path,
                "company": row["Company"],
                "roll": row["Roll number"]
            }
            csv_docs.append(Document(page_content=content, metadata=metadata))

        # Read the text content from the file
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        pdf_docs = text_splitter.split_documents(documents)

        all_docs = csv_docs + pdf_docs
        # Display information about the split documents
        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(all_docs)}")
        print(f"Sample chunk:\n{all_docs[0].page_content}\n")

        # Create the vector store and persist it automatically
        print("Creating vector store")
        db = FAISS.from_documents(all_docs, embeddings)
        db.save_local("faiss_index")
        print("\n--- Finished creating vector store ---")

    else:
        print("Vector store already exists. No need to initialize.")
        # Load the existing vector store with the embedding function
        #db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        db = FAISS.load_local("faiss_index", embeddings)

    return db
