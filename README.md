# PolicyGPT - AI-Powered Insurance Policy Assistant

An intelligent system that uses Large Language Models (LLMs) to process natural language queries and retrieve relevant information from insurance policy documents using semantic search.

## 🎯 Problem Statement

Build a system that uses Large Language Models (LLMs) to process natural language queries and retrieve relevant information from large unstructured documents such as policy documents, contracts, and emails.

### Objective
The system takes input queries like:
- "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
- "What is deductible or co-payment?"
- "My father had liver surgery, can we get reimbursement?"

And provides structured responses with:
- Decision (Approved/Rejected/Partial/Need more info)
- Justification based on policy clauses
- Amount covered (if applicable)
- Next steps for the user

## 🚀 Features

- **Semantic Search**: Uses Sentence Transformers to find relevant policy clauses from 222+ document chunks
- **AI-Powered Analysis**: Leverages Gemini API for intelligent query understanding and response generation
- **Modern UI**: Beautiful, responsive web interface with 3D effects and animations
- **Real-time Processing**: Instant results with loading animations and progress indicators
- **Multi-language Support**: Handles queries in English, Hindi, and mixed languages

## 🛠️ Technology Stack

- **Backend**: Python, Flask, Sentence Transformers
- **Frontend**: HTML5, CSS3, JavaScript
- **AI/ML**: ChromaDB, Sentence Transformers, Gemini API
- **Document Processing**: Custom chunking system for policy documents

## 📁 Project Structure

```
llm-model/
├── app.py                          # Flask backend server
├── policy_assistant_ui.html        # Frontend UI
├── hackrx_full_demo.py            # Standalone demo script
├── build_vector_store.py          # Vector database builder
├── query_vector_store.py          # Query script
├── chunk_documents.py             # Document chunking utility
├── requirements.txt               # Python dependencies
├── README.md                     # This file
├── data/                         # Source policy documents
│   ├── doc1.txt
│   ├── doc2.txt
│   └── doc3.txt
├── chunks/                       # Processed document chunks (222 files)
└── chromadb_store/              # Vector database storage
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ankitrraj/insurance_bot
   cd policygpt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   Navigate to `http://localhost:5000`

## 💡 Usage Examples

### Example Queries:
- "What is deductible or co-payment?"
- "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
- "My father had liver surgery, can we get reimbursement?"
- "bhai mujhe accident ho gaya hai ilaaz ke liye 50000 ruppes chaiye"

### Sample Response:
```json
{
  "decision": "Need more info",
  "justification": "Based on the policy clauses, deductibles and co-payments are standard cost-sharing mechanisms. A deductible is a fixed amount you pay before insurance coverage begins, while a co-payment is a percentage of each claim.",
  "amount": "N/A (depends on specific policy terms)",
  "next_steps": "Check your specific policy document for the exact deductible amount and co-payment percentage applicable to your plan."
}
```

## 🔧 API Endpoints

### POST /api/query
Process a natural language query and return relevant policy information.

**Request:**
```json
{
  "query": "What is deductible or co-payment?"
}
```

**Response:**
```json
{
  "success": true,
  "matches": [
    {
      "rank": 1,
      "score": 0.8269,
      "source": "doc1.txt_chunk10.txt",
      "content": "Deductible refers to the fixed amount..."
    }
  ],
  "ai_response": {
    "decision": "Approved",
    "justification": "...",
    "amount": "N/A",
    "next_steps": "..."
  }
}
```

## 🎨 UI Features

- **Starry Background**: Animated star field for premium feel
- **3D Effects**: Floating document with hover animations
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Loading Animations**: Visual feedback during processing
- **Modern Typography**: Clean, readable fonts
- **Gradient Accents**: Professional color scheme

## 🔍 How It Works

1. **Document Processing**: Policy documents are chunked into smaller, manageable pieces
2. **Vector Embeddings**: Each chunk is converted to a vector using Sentence Transformers
3. **Query Processing**: User queries are embedded and compared with document chunks
4. **Semantic Search**: Top 5 most relevant chunks are retrieved
5. **AI Analysis**: Gemini API analyzes the chunks and generates structured responses
6. **Result Display**: Results are presented in a clean, organized format

## 🏆 HackRx Hackathon Submission

This project was developed for the HackRx hackathon, demonstrating:
- Advanced NLP techniques for document understanding
- Real-world application in insurance domain
- Modern web development with AI integration
- Scalable architecture for document processing

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

- **Developer**: [ankit rajoria]
- **Email**: [ankitrajoria81@gmail.com]
- **GitHub**: [@ankitrraj]

---

⭐ Star this repository if you found it helpful! 