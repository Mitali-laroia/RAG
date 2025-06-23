# 📚 PDF RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) application built with Streamlit that allows you to upload PDF documents and ask questions about their content using OpenAI's GPT models and Qdrant vector database.

## 🌟 Features

- **PDF Upload & Processing**: Upload any PDF document and automatically extract and chunk the text
- **Intelligent Chunking**: Uses RecursiveCharacterTextSplitter for optimal text segmentation
- **Vector Search**: Leverages Qdrant Cloud for fast and accurate similarity search
- **AI-Powered Q&A**: Uses OpenAI's GPT-4o-mini for generating contextual responses
- **Chat History**: Maintains conversation history for better user experience
- **Page References**: Provides page numbers for easy reference back to source material
- **Cloud-Ready**: Fully configured for deployment on Streamlit Cloud

## 🚀 Live Demo

[🔗 Try the app here](https://raggit-7kvxk6pbcz7fq7gl2bhzpn.streamlit.app/)

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o-mini
- **Vector Database**: Qdrant Cloud
- **Embeddings**: OpenAI text-embedding-3-large
- **Text Processing**: LangChain
- **PDF Processing**: PyPDF

## 📋 Prerequisites

Before running this application, you need:

1. **OpenAI API Key**: Get it from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Qdrant Cloud Account**: Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
3. **Python 3.8+**: Make sure you have Python installed

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Mitali-laroia/RAG.git
cd pdf-rag-assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
QDRANT_URL = "https://your-cluster-id.eu-central.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your-qdrant-api-key-here"
```

**⚠️ Important**: Add `.streamlit/secrets.toml` to your `.gitignore` file to avoid committing sensitive information.

## 🎯 Usage

### Running Locally

```bash
streamlit run streamlit.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

1. **Upload PDF**: Click "Choose a PDF file" and select your document
2. **Process Document**: Click "🚀 Process PDF" to create embeddings
3. **Start Chatting**: Click "💬 Start Chatting!" to begin asking questions
4. **Ask Questions**: Type your questions about the document content
5. **View Responses**: Get AI-powered answers with page references

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. **Push to GitHub**: Make sure your code is in a GitHub repository
2. **Visit Streamlit Cloud**: Go to [share.streamlit.io](https://share.streamlit.io)
3. **Deploy**: Click "New app" and connect your repository
4. **Configure Secrets**: Add your API keys in the Streamlit Cloud secrets section:

```toml
OPENAI_API_KEY = "your-openai-api-key"
QDRANT_URL = "https://your-cluster-id.eu-central.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your-qdrant-api-key"
```

### Deploy to Other Platforms

The app can also be deployed to:
- Heroku
- AWS EC2
- Google Cloud Platform
- Azure Container Instances

## 📁 Project Structure

```
pdf-rag-assistant/
│
├── streamlit.py              # Main application file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── README.md                # This file
│
├── .streamlit/
│   └── secrets.toml         # Local secrets (not committed)
│
└── docs/
    └── screenshots/         # App screenshots (optional)
```

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | ✅ |
| `QDRANT_URL` | Qdrant Cloud cluster URL | ✅ |
| `QDRANT_API_KEY` | Qdrant Cloud API key | ✅ |

## 🎨 Features in Detail

### PDF Processing
- Supports various PDF formats
- Automatic text extraction and cleaning
- Intelligent document chunking with overlap
- Preserves page number metadata

### Vector Search
- Uses OpenAI's latest embedding model (text-embedding-3-large)
- Qdrant Cloud for scalable vector storage
- Similarity search with configurable results count
- Efficient retrieval of relevant document sections

### AI Responses
- Context-aware responses using GPT-4o-mini
- Page number references for source verification
- Maintains conversation context
- Error handling and fallback responses

### User Experience
- Clean, intuitive interface
- Real-time processing feedback
- Chat history with expandable Q&A pairs
- Easy document switching
- Mobile-responsive design

## 🚨 Troubleshooting

### Common Issues

**Error: "qdrant_config is not defined"**
- Ensure your Qdrant URL and API key are properly set in secrets
- Check that your Qdrant cluster is running and accessible

**Error: "OpenAI API key not found"**
- Verify your OpenAI API key is correctly set
- Check that you have sufficient API credits

**PDF Processing Fails**
- Ensure the PDF is not password-protected
- Check that the PDF contains extractable text (not just images)
- Verify the file size is within limits

### Getting Help

1. Check the [Issues](https://github.com/Mitali-laroia/RAG/issues) page
2. Review the troubleshooting section above
3. Create a new issue with detailed error information

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web app framework
- [OpenAI](https://openai.com/) for the powerful language models
- [Qdrant](https://qdrant.tech/) for the vector database solution
- [LangChain](https://langchain.com/) for the document processing tools

## 📊 Roadmap

- [ ] Support for multiple file formats (Word, TXT, etc.)
- [ ] Advanced search filters and options
- [ ] Document comparison features
- [ ] User authentication and document management
- [ ] Batch processing capabilities
- [ ] Custom embedding models support
- [ ] Multi-language support

## 📞 Support

If you found this project helpful, please give it a ⭐ on GitHub!

For support, email 04042001mitali@gmail.com or create an issue in the repository.

---

**Made with ❤️ by [Your Name](https://github.com/Mitali-laroia)**