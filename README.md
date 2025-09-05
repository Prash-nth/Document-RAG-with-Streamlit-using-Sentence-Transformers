Document RAG with Streamlit
A Streamlit-based application for Retrieval-Augmented Generation (RAG) using Sentence Transformers. This project allows users to upload PDF documents or provide web URLs, process their content, and query the documents to retrieve relevant information based on semantic similarity.
Features

Document Upload: Upload PDF files or provide URLs to process and store document content.
Text Embedding: Uses the all-MiniLM-L6-v2 Sentence Transformer model to create embeddings for document chunks.
Similarity Search: Retrieves relevant document sections using cosine similarity with a configurable threshold.
Interactive UI: Built with Streamlit, featuring a user-friendly interface with a sidebar for configuration and a chat-like query system.
Source Tracking: Displays metadata (e.g., file name or URL) for retrieved documents, including similarity scores.
Clear History: Option to reset the chat history and document store via the sidebar.

Prerequisites

Python 3.8 or higher
Git (for cloning the repository)
A GitHub account (to host or contribute to the repository)

Installation

Clone the Repository:
git clone https://github.com/your-username/document-rag-streamlit.git
cd document-rag-streamlit


Create a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

The requirements.txt includes:

streamlit==1.31.0
sentence-transformers==2.3.1
langchain==0.1.0
langchain-community==0.0.16
scikit-learn==1.4.0
PyPDF2==3.0.1
beautifulsoup4==4.12.2
numpy==1.26.3


Run the Application:
streamlit run document_rag.py

This will launch the app in your default browser (typically at http://localhost:8501).


Usage

Upload Documents:

Use the sidebar to upload a PDF file or enter a URL.
The app processes the document, splits it into chunks (1000 characters with 200-character overlap), and stores embeddings using the Sentence Transformer model.


Configure Search:

Adjust the Similarity Threshold in the sidebar (0.0 to 1.0, default 0.5) to control the strictness of document retrieval. Lower values return more documents but may include less relevant ones.


Query Documents:

Enter a query in the chat input box at the bottom of the page.
The app retrieves up to 3 relevant document chunks based on cosine similarity and displays a response with the most relevant content.
Expand the "See document sources" section to view the retrieved document chunks, their sources, and similarity scores.


Clear History:

Click the "Clear Chat History" button in the sidebar to reset the chat history, document store, and embedder.



Example

Upload a PDF named sample.pdf or enter a URL like https://example.com.
Wait for the app to process and store the document (you’ll see a success message).
Ask a question like, "What is the main topic of the document?"
The app responds with relevant text and source information, e.g.:The document discusses machine learning techniques for text processing.
Source: sample.pdf (Similarity: 0.92)



Project Structure
document-rag-streamlit/
├── document_rag.py      # Main Streamlit application
├── requirements.txt     # Project dependencies
├── README.md           # This file
├── LICENSE             # MIT License (optional)
└── .gitignore          # Ignores Python cache, virtual env, etc.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make changes and commit (git commit -m "Add your feature").
Push to your fork (git push origin feature/your-feature).
Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes relevant tests.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Troubleshooting

PDF Processing Errors: Ensure the uploaded PDF is not corrupted or password-protected.
Web URL Errors: Verify the URL is accessible and contains parseable content. Some websites may block scraping.
Dependency Issues: Confirm all dependencies are installed correctly. If errors persist, try upgrading pip (pip install --upgrade pip) or recreating the virtual environment.
Performance: Processing large documents or URLs may take time due to embedding generation. Consider reducing the chunk size in document_rag.py for faster processing.

Future Improvements

Add support for more document formats (e.g., DOCX, TXT).
Integrate a more advanced response generation model (e.g., using xAI's Grok API).
Implement persistent storage for documents and embeddings.
Add export functionality for retrieved results.

Contact
For questions or feedback, open an issue on the GitHub repository or contact [your.email@example.com].
