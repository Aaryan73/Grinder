# Study Partner Finder 🎓

An intelligent study group matching system that uses ML-powered algorithms to connect university students with compatible study partners. The system matches students based on their course, topics of interest, knowledge level, and desired group size while optimizing for group compatibility and learning effectiveness.

![Image](gif.gif)

## 🌟 Features

- **Smart Matching Algorithm**: Utilizes multiple embedding models (SBERT, Cohere, Domain-specific) to find the most compatible study partners
- **Multi-factor Compatibility**: Considers course alignment, topic similarity, knowledge level balance, and preferred group size
- **Real-time Chat Interface**: Easy registration and group formation through an intuitive chat interface
- **Group Status Tracking**: Monitor your group formation progress and view detailed compatibility metrics
- **Optimized Group Formation**: Ensures balanced knowledge distribution and topic alignment within groups
- **Interactive Dashboard**: View your profile, group details, and matching metrics through a clean sidebar interface

## 🏗️ Architecture

The application is built using a modern tech stack:

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **ML Components**:
  - Sentence-BERT (SBERT)
  - Cohere Embeddings
  - Custom Domain-Specific Models
  - Scikit-learn for similarity metrics

## 🚀 Installation

1. Clone the repository:
```bash
git https://github.com/Aaryan73/Grinder.git
cd Grinder
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file
COHERE_API_KEY=your_cohere_api_key
API_BASE_URL=http://localhost:8000
```

## 🔧 Configuration

The system uses several configuration parameters that can be adjusted in the matching algorithm:

```python
MIN_KNOWLEDGE_AVERAGE = 2.0
MIN_TOPIC_SIMILARITY = 0.75
MIN_INDIVIDUAL_SIMILARITY = 0.70
MAX_SIMILARITY_VARIANCE = 0.15
```

These parameters can be tuned to adjust the strictness of group matching criteria.

## 🖥️ Usage

1. Start the FastAPI backend:
```bash
uvicorn api:app --reload
```

2. Launch the Streamlit frontend:
```bash
streamlit run main2.py
```

3. Access the application at `http://localhost:8501`

## 💡 How It Works

### Registration Process
1. Students enter their details through a chat interface:
   - Name
   - Contact number
   - Course code
   - Topics they need help with
   - Current knowledge level
   - Desired group size

### Matching Algorithm
The system uses a sophisticated matching algorithm that:
1. Generates embeddings for student topics using multiple models
2. Calculates weighted similarity scores between potential group members
3. Evaluates knowledge level distribution within groups
4. Optimizes for both topic similarity and knowledge balance
5. Ensures minimum compatibility thresholds are met

### Group Formation
- Groups are formed progressively as compatible students join
- Real-time updates on group status
- Detailed metrics on group compatibility
- Automatic group completion notifications

## 📊 Matching Metrics

The system provides detailed metrics for each group:
- Knowledge Balance Score
- Topic Similarity Score
- Overall Compatibility Score
- Individual Similarity Scores between members
- Similarity Distribution Statistics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SentenceTransformers team for their amazing SBERT implementation
- Cohere for their powerful embedding models
- Streamlit team for their awesome frontend framework
- FastAPI team for their high-performance backend framework