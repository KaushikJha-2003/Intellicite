# Intellicite: AI Document Assistant 💡

Intellicite lets you upload PDFs, websites, or YouTube videos and interact with them through natural conversations.  
It uses Google Gemini, LangChain, HuggingFace embeddings, and ChromaDB for retrieval-augmented generation (RAG).  
Built with Streamlit for an intuitive chat interface and customizable AI personas.  

## 🚀 Features
- Upload and chat with **PDF documents**  
- Extract knowledge from **websites**  
- Interact with **YouTube video transcripts**  
- Choose AI personas: Helpful Assistant, Technical Expert, Business Analyst, or ELI5  
- Reset and manage ChromaDB storage easily  

## 🛠️ Tech Stack
- **Frontend**: Streamlit  
- **LLM**: Google Gemini via `langchain_google_genai`  
- **Embeddings**: HuggingFace MiniLM-L6-v2  
- **Vector Store**: ChromaDB  
- **Document Loaders**: PyPDFLoader, WebBaseLoader, YoutubeLoader  

## 📂 Project Setup
1. Clone the repo:  
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Add your **Google API Key** to a `.env` file:  
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```
4. Run the app:  
   ```bash
   streamlit run app.py
   ```

## 🤝 Contributing
Feel free to open issues or submit PRs to improve the project.

---
*Powered by Google Gemini & LangChain*
