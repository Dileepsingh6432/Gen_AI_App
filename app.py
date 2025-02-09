import os
import streamlit as st
# from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain.llms import Ollama

# With these updated imports
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# llm = Ollama(model="deepseek-r1:1.5b")
# llm.invoke('Hi there! who are you')
# llm.invoke('Is Taiwan a sovereign country?')
# Step 1: Load and preprocess documents
def load_and_split_documents(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    return texts
# texts = load_and_split_documents('test.txt')
# texts

# Step 2: Create embeddings and FAISS vector store
def create_vector_store(texts):
    # Use a pre-trained embedding model (e.g., Sentence Transformers)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    
    return vector_store

# vector_store=create_vector_store(texts)
# vector_store

# Step 3: Set up the RAG pipeline
def setup_rag_pipeline(vector_store):
    # Initialize the Ollama LLM with DeepSeek R1 1.5B
    llm = Ollama(model="deepseek-r1:1.5b")
    
    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain

# qa_chain =setup_rag_pipeline(vector_store)
# qa_chain

# Step 4: Query the RAG pipeline
def query_rag_pipeline(qa_chain, query):
    # result = qa_chain({"query": query})
    result = qa_chain.invoke({"query": query})

    return result["result"], result["source_documents"]

# query = 'who is Dileep?'
# result,source_docs = query_rag_pipeline(qa_chain, query)
# print(result)
# print(result)
# print(source_docs)

# Streamlit UI
def main():
    st.title("RAG Chatbot with DeepSeek R1 1.5B")
    
    # Load and process documents
    file_path = "test.txt"  # Replace with your document path
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return
    
    texts = load_and_split_documents(file_path)
    vector_store = create_vector_store(texts)
    qa_chain = setup_rag_pipeline(vector_store)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    st.subheader("Chat History")
    for i, (user_query, bot_response) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {user_query}")
        st.markdown(f"**Bot:** {bot_response}")
        st.markdown("---")
    
    # User input
    user_query = st.text_input("Ask a question:")
    
    if user_query:
        # Query the RAG pipeline
        bot_response, source_docs = query_rag_pipeline(qa_chain, user_query)
        
        # Update chat history (keep only last 3 interactions)
        st.session_state.chat_history.append((user_query, bot_response))
        if len(st.session_state.chat_history) > 3:
            st.session_state.chat_history.pop(0)
        
        # Display the bot's response
        st.subheader("Bot's Response")
        st.markdown(bot_response)
        
        # Display source documents
        st.subheader("Source Documents")
        for doc in source_docs:
            st.markdown(doc.page_content)
            st.markdown("---")
        
        # Rerun to update the chat history display
        st.rerun()


if __name__ == "__main__":
    main()
