# LangChain Document Q&A System

A simple yet powerful document question-answering system built with LangChain that scrapes web content and enables intelligent querying using OpenAI's language models.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project demonstrates a complete RAG (Retrieval-Augmented Generation) pipeline using LangChain. It scrapes documentation from websites, processes the content into searchable chunks, creates vector embeddings, and enables users to ask questions about the content using natural language.

The system uses OpenAI's GPT models for answer generation and FAISS for efficient vector storage and retrieval, with LangSmith integration for monitoring and tracing.

## ✨ Features

- **Web Content Scraping**: Automatically scrape and process documentation from websites
- **Document Chunking**: Intelligent text splitting with configurable chunk sizes and overlap
- **Vector Embeddings**: Convert text chunks into searchable vector representations using OpenAI embeddings
- **FAISS Vector Store**: Efficient similarity search and retrieval using Facebook's FAISS library
- **Question Answering**: Natural language queries with context-aware responses
- **LangSmith Integration**: Built-in tracing and monitoring for debugging and optimization
- **Retrieval Chain**: Complete RAG pipeline from query to contextual answer generation

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- OpenAI API key
- LangSmith API key (optional, for tracing)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/angela41017/LangChain.git
   cd LangChain
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## 🔧 Configuration

Create a `.env` file in the root directory with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith Configuration (Optional - for tracing and monitoring)
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_project_name
```

## 📁 Project Structure

```
LangChain/
├── 1.1-openai/
│   ├── 1.1.2-Simpleapp.ipynb    # Main notebook with complete RAG implementation
│   └── requirements.txt          # Python dependencies
├── .env                         # Environment variables (create from .env.example)
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## 📖 Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**
   Navigate to `1.1-openai/1.1.2-Simpleapp.ipynb`

3. **Run the cells sequentially**
   The notebook contains a complete workflow:
   - Environment setup and API key configuration
   - Web content scraping using WebBaseLoader
   - Document chunking with RecursiveCharacterTextSplitter
   - Vector embedding creation with OpenAI embeddings
   - FAISS vector store creation and querying
   - Retrieval chain setup for question answering

### Basic Python Script Usage

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Load and process documents
loader = WebBaseLoader("https://docs.smith.langchain.com/administration/tutorials/manage_spend")
docs = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Set up the LLM and prompt
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:
<context>
{context}
</context>
""")

# Create the retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Ask questions
response = retrieval_chain.invoke({"input": "What are LangSmith's usage limits?"})
print(response['answer'])
```

## 🧪 Examples

### Example 1: Web Document Processing

The main notebook demonstrates loading content from LangSmith documentation:

```python
# Load web content
loader = WebBaseLoader("https://docs.smith.langchain.com/administration/tutorials/manage_spend")
docs = loader.load()

# Process into searchable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
```

### Example 2: Vector Search

```python
# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Perform similarity search
query = "LangSmith usage limits"
results = vectorstore.similarity_search(query)
print(results[0].page_content)
```

### Example 3: Question Answering

```python
# Set up retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Ask questions about the content
response = retrieval_chain.invoke({
    "input": "How can I optimize LangSmith spending?"
})
print(response['answer'])
```

### Example Queries You Can Try

Based on the loaded LangSmith documentation, you can ask questions like:
- "What are the two usage limits in LangSmith?"
- "How can I reduce LangSmith costs?"
- "What is extended data retention?"
- "How do I set usage limits per workspace?"

## 🧪 Testing

Currently, the project is implemented as a Jupyter notebook for experimentation and learning. To test the functionality:

1. Run all cells in the notebook sequentially
2. Try different queries with the retrieval chain
3. Experiment with different chunk sizes and overlap values
4. Test with different websites by changing the WebBaseLoader URL

## 🛠 Dependencies

The project uses the following key libraries:

- **langchain**: Core LangChain framework
- **langchain-openai**: OpenAI integrations for LLMs and embeddings
- **langchain-community**: Community document loaders and vector stores
- **langchain-core**: Core LangChain components
- **faiss-cpu**: Facebook's similarity search library
- **python-dotenv**: Environment variable management
- **beautifulsoup4**: Web scraping support
- **ipykernel**: Jupyter notebook support

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [FAISS Documentation](https://faiss.ai/)
- [LangChain Community Loaders](https://python.langchain.com/docs/integrations/document_loaders/)

## 🔧 Customization

You can easily customize this project by:

1. **Changing the data source**: Replace the WebBaseLoader URL with any website or use different loaders for PDFs, CSVs, etc.
2. **Adjusting chunk parameters**: Modify `chunk_size` and `chunk_overlap` in the text splitter
3. **Using different models**: Replace "gpt-4o" with other OpenAI models or different providers
4. **Experimenting with prompts**: Customize the prompt template for different response styles
5. **Adding different vector stores**: Replace FAISS with Chroma, Pinecone, or other vector databases

## 🚀 Next Steps

To extend this project, consider:

- Converting the notebook into a web application using Streamlit or Gradio
- Adding support for multiple document sources
- Implementing conversation memory for multi-turn dialogues
- Adding evaluation metrics for answer quality
- Creating a REST API for the Q&A system
- Implementing advanced retrieval techniques like hybrid search

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the amazing framework
- [OpenAI](https://openai.com/) for the powerful language models and embeddings
- [Facebook Research](https://github.com/facebookresearch/faiss) for the FAISS vector search library
- [LangSmith](https://smith.langchain.com/) for monitoring and tracing capabilities

---

**Note**: This is a learning project demonstrating basic RAG implementation. For production use, consider adding error handling, input validation, and proper testing.
