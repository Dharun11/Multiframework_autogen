# MultiAgent framework_autogen


## Overview
A sophisticated search assistant leveraging multi-agent architecture, semantic chunking, and retrieval-augmented generation (RAG) to provide intelligent, context-aware search capabilities across Wikipedia and a custom vector store.

## Responses
![App Interface](/assets/interface.png)
![Vector Store Setup](/assets/rag_response.png)
![Search Results](/assets/wiki_response.png)

## Features
- Multi-agent routing system
- Semantic text chunking
- Vector store with embeddings
- Wikipedia and custom document search
- Streamlit web interface

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/multi-agent-rag-assistant.git
cd multi-agent-rag-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key
```

## Usage
```bash
streamlit run app.py
```

### Workflow
1. Setup vector store via sidebar button
2. Enter query in search box
3. Click "Search" to retrieve results

## Components
- **Semantic Chunking**: Intelligently splits documents
- **Embedding Generation**: Uses Hugging Face embeddings
- **Routing Agent**: Directs queries to appropriate knowledge base
- **Retrieval Agents**: 
  - Wikipedia search
  - Vector store retrieval

## Technologies
- Streamlit
- Hugging Face Transformers
- ChromaDB
- Groq API
- Autogen

## Contributing
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

