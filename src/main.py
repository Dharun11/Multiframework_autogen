
import os
import re
import streamlit as st
import requests
import wikipedia
import chromadb
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AzureOpenAI
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent
from typing import Annotated, List
import torch
from transformers import AutoTokenizer, AutoModel
from groq import Groq

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load environment variables
load_dotenv()


api_key = os.getenv('GROQ_API_KEY')



config_list = {
    "model": "llama-3.1-70b-versatile",

    "api_type": "groq",
    "api_key": api_key,
}

class HuggingFaceEmbedding:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize Hugging Face embedding model
        
        Args:
            model_name (str): Hugging Face model identifier
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
    def generate_embedding(self, texts):
        """
        Generate embeddings for input texts
        
        Args:
            texts (list): List of text chunks
        
        Returns:
            list: List of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize inputs
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
            # Mean pooling
            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()

def semantic_chunking(text: str, num_chunks: int) -> List[str]:
    """
    Perform semantic chunking using TF-IDF and KMeans clustering
    
    Args:
        text (str): Input text to be chunked
        num_chunks (int): Number of semantic chunks to create
    
    Returns:
        List[str]: List of semantic chunks
    """
    # Split text into sentences using a regex for punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Convert sentences into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_chunks, random_state=42)
    kmeans.fit(sentence_vectors)
    
    # Group sentences into chunks based on their cluster label
    chunks = [[] for _ in range(num_chunks)]
    for i, label in enumerate(kmeans.labels_):
        chunks[label].append(sentences[i])
    
    # Combine sentences in each cluster into a single chunk
    return [' '.join(chunk) for chunk in chunks]

def split_large_chunks(chunks: List[str], max_length: int, split_factor: int) -> List[str]:
    
    result_chunks = []

    for chunk in chunks:
        if len(chunk) > max_length:
            # Perform further chunking on the large chunk
            sub_chunks = semantic_chunking(chunk, split_factor)
            result_chunks.extend(
                split_large_chunks(sub_chunks, max_length, split_factor)
            )
        else:
            result_chunks.append(chunk)

    return result_chunks

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    
    embedder = HuggingFaceEmbedding()
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Generate embedding for each chunk
            embedding = embedder.generate_embedding(chunk)
            embeddings.append(embedding[0])  # Hugging Face returns a list of embeddings
            st.write(f"Completed embedding for chunk {i}")
        except Exception as e:
            st.error(f"Error generating embedding for chunk {i}: {e}")
    
    return embeddings

def fetch_url_content(url):
    """Fetch content from a given URL"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.get_text(separator="\n")
    return content

def preprocess_chunks(docs):
    """Preprocess and clean document text"""
    cleaned_docs = docs.replace('\n', ' ')
    cleaned_docs = re.sub(r'\s+', ' ', cleaned_docs)
    cleaned_docs = re.sub(r'[^\w\s.,!?]', '', cleaned_docs)
    cleaned_docs = cleaned_docs.strip()
    cleaned_docs = re.sub(r'\n\s*\n', '\n', cleaned_docs)
    
    return cleaned_docs

def search_wiki(query: str):
    """Search Wikipedia and return content"""
    try:
        result = wikipedia.page(query).content
    except Exception as e:
        st.error(f"An error occurred while searching Wikipedia: {e}")
        result = "No Wikipedia content found."
    
    return result[:10000]

def retrieve_from_chroma(query: Annotated[str, "The query to search in the vector store"]):
    """Retrieve content from Chroma vector store"""
    chroma_client = chromadb.PersistentClient(path='vectorstore')
    collection = chroma_client.get_collection(name="Multiagent1_rag")
    
    # Create Azure OpenAI client
    
    
    # Generate query embedding
    query_embedding = generate_embeddings(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,  # Increased to get more context
    )
    
    # Combine retrieved documents
    retrieved_docs = results["documents"][0]
    return " ".join(retrieved_docs) if retrieved_docs else "No relevant content found."

def initialize_agents():
    """Initialize agents for routing and searching"""
    route_query = AssistantAgent(
        name="route_query",
        system_message="""You are an expert at routing a user question to a vectorstore or wikipedia.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. Otherwise, say wiki-search.
        you have to answer either Vectorstore or wiki-search in one word
        for example:
        User query is who is sharukhan 
        Answer: Wiki-search""",
        llm_config=config_list,
        human_input_mode="NEVER"
    )

    user_proxy = AssistantAgent(
        name="user_proxy",
        system_message="Given a user question choose to route it to wikipedia or a vectorstore.",
        llm_config=config_list
    )

    wiki_agent = ConversableAgent(
        name="wiki_agent",
        llm_config=config_list,
        system_message="You are a helpful AI assistant with access to a knowledge base. "
        "You use the wiki-search to retrieve context for user queries and generate responses. "
        "Always combine retrieved knowledge with your understanding to provide accurate answers."
    )

    rag_agent = ConversableAgent(
        name="RAGAgent",
        system_message=(
            "You are a helpful AI assistant with access to a knowledge base. "
            "You use the vector store to retrieve context for user queries and generate responses. "
            "Always combine retrieved knowledge with your understanding to provide accurate answers."
        ),
        llm_config=config_list,
    )

    wiki_agent.register_for_llm(name="search_wiki", description="A function used to retrieve the relevant content from wiki search")(search_wiki)
    user_proxy.register_for_execution(name="search_wiki")(search_wiki)

    rag_agent.register_for_llm(name="retrieve_from_chroma", description="A function used to retrieve the relevant content from vector db")(retrieve_from_chroma)
    user_proxy.register_for_execution(name="retrieve_from_chroma")(retrieve_from_chroma)

    return route_query, user_proxy, wiki_agent, rag_agent

def final_output(query: str, route_query, user_proxy, wiki_agent, rag_agent):
    """Process the query and return the appropriate response"""
    res_method = user_proxy.initiate_chat(route_query, message=query, max_turns=1)
    
    if res_method.summary == "Wiki-search":
        wiki_result = user_proxy.initiate_chat(wiki_agent, message=query, max_turns=2)
        return wiki_result.summary,res_method.summary
    else:
        vector_res = user_proxy.initiate_chat(rag_agent, message=query, max_turns=2)
        return vector_res.summary,res_method.summary

def setup_vectorstore():
    
    try:
        # URLs to fetch content from
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]

        # Fetch and combine documents
        doc = ""
        for url in urls:
            doc += fetch_url_content(url)

        # Preprocess the document
        cleaned_doc = preprocess_chunks(doc)

        # Semantic Chunking
        chunks = semantic_chunking(cleaned_doc, num_chunks=30)
        
        # Split large chunks
        sorted_chunks = split_large_chunks(chunks, max_length=8000, split_factor=5)

        # Create Chroma client and collection
        chroma_client = chromadb.PersistentClient(path='vectorstore')
        
        # Check if collection already exists
        try:
            chroma_client.get_collection(name="Multiagent1_rag")
            st.info("Vector store already exists.")
            return
        except Exception:
            # Create collection if it doesn't exist
            new_collection = chroma_client.create_collection(name="Multiagent1_rag")
            
            # Generate embeddings using Hugging Face
            embeddings = generate_embeddings(sorted_chunks)
            
            # Add chunks and embeddings to the collection
            new_collection.add(
                documents=sorted_chunks,  
                embeddings=embeddings,    
                ids=[f"chunk_{i}" for i in range(len(sorted_chunks))],  
                metadatas=[{"chunk_index": i} for i in range(len(sorted_chunks))]  
            )
            
            st.success(f"Vector store created successfully with {len(sorted_chunks)} semantic chunks.")

    except Exception as e:
        st.error(f"Error setting up vector store: {e}")

def main():
    st.title("Multi-Agent RAG Search Assistant")
    
    # Sidebar for setup and configuration
    st.sidebar.header("Configuration")
    
    # Vector Store Setup
    if st.sidebar.button("Setup Vector Store"):
        setup_vectorstore()
    
    # Initialize agents
    route_query, user_proxy, wiki_agent, rag_agent = initialize_agents()
    
    # Query input
    query = st.text_input("Enter your query:", "")
    
    # Search button
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                try:
                    result,res_method= final_output(query, route_query, user_proxy, wiki_agent, rag_agent)
                    st.success(f"Search Result:{res_method}")
                    st.write(result)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()


