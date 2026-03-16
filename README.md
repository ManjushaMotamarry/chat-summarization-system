# 💬 AI-Powered Chat Summarization System

An end-to-end MLOps pipeline that automatically generates concise summaries of customer support conversations to reduce agent onboarding time during handoffs.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120.2-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)](https://streamlit.io/)

---

## 🎯 Project Overview

**Problem:** Customer service agents spend 2-3 minutes reading through lengthy chat histories when taking over conversations from chatbots or other agents.

**Solution:** AI-powered summarization that condenses conversations from ~80 words to ~20 words, reducing agent onboarding time to seconds.

**Business Impact:**
- ⏱️ Saves 2+ minutes per handoff
- 📈 Agents handle 20-30% more tickets per hour
- 😊 Faster response times improve customer satisfaction
- 💰 Reduces operational costs

---

## 🏗️ System Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Source   │────▶│  Prefect Pipeline │────▶│  SQLite Database│
│  (HuggingFace)  │     │  (Orchestration)  │     │  (3 Tables)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit UI   │◀────│   FastAPI Server │◀────│  BART Model     │
│  (Demo)         │     │   (REST API)     │     │  (Fine-tuned)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 📊 Model Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROUGE-1** | 0.477 | 47.7% word overlap with human summaries |
| **ROUGE-2** | 0.246 | 24.6% phrase-level match |
| **ROUGE-L** | 0.403 | 40.3% structural similarity |
| **Compression** | 4.2x | Reduces text to ~25% of original length |

**Comparison to State-of-the-Art:**
- Published SOTA on SAMSum: ROUGE-L ~0.49
- This implementation: ROUGE-L 0.403 (82% of SOTA)
- **Competitive baseline** achieved with 3 epochs of training

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 8GB+ RAM
- ~2GB disk space for model

### Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd chat-summarization-system
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Run the Demo

**Option 1: Streamlit Dashboard (Recommended for testing)**
```bash
streamlit run dashboards/summarization_app.py
```
Access at: `http://localhost:8501`

**Option 2: FastAPI Server (Recommended for integration)**
```bash
python src/api/summarization_api.py
```
API docs at: `http://localhost:8000/docs`

---

## 📁 Project Structure
```
chat-summarization-system/
├── configs/                      # Configuration files
│   ├── dataset_config.yaml       # Dataset settings
│   ├── preprocessing_config.yaml # Text preprocessing rules
│   └── model_config.yaml         # Model training parameters
├── data/
│   ├── processed/                # Train/val/test splits
│   └── chat_conversations.db     # SQLite database
├── src/
│   ├── data/
│   │   ├── database.py           # SQLAlchemy models
│   │   ├── dataset_loader.py     # HuggingFace data loader
│   │   ├── text_preprocessor.py  # Smart text cleaning
│   │   └── dataset_preparation.py# Train/val/test splitting
│   ├── models/
│   │   └── summarization_trainer.py  # Model training pipeline
│   ├── api/
│   │   └── summarization_api.py  # FastAPI service
│   └── utils/
│       ├── config_loader.py      # Configuration management
│       └── logger.py             # Centralized logging
├── prefect_flows/
│   └── chat_summarization_pipeline.py  # Data pipeline orchestration
├── dashboards/
│   └── summarization_app.py      # Streamlit demo interface
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # EDA notebook
├── models/
│   └── best_model/               # Trained BART model (not in git)
├── scripts/                      # Utility scripts
└── tests/                        # Unit tests
```

---

## 🔄 Complete Pipeline Workflow

### 1. Data Ingestion & Processing
```bash
python prefect_flows/chat_summarization_pipeline.py
```
**What it does:**
- Downloads SAMSum dataset (14,731 conversations)
- Loads into SQLite database
- Applies smart preprocessing (preserves context with tokens like [photo], [link])
- Validates data quality with Great Expectations
- Generates pipeline report

### 2. Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```
**Key findings:**
- Average conversation: 83 words, 11 messages
- Average summary: 20 words
- Target compression: 4.2x
- Distribution: Right-skewed (most conversations are short)

### 3. Model Training
```bash
python src/models/summarization_trainer.py
```
**Training details:**
- Model: `facebook/bart-base` (139M parameters)
- Dataset: 11,784 train / 1,473 val / 1,474 test (stratified by length)
- Training time: ~2-3 hours on CPU (M-series Mac)
- Tracking: MLflow (`mlruns/` directory)

### 4. Model Testing
```bash
python scripts/test_model_inference.py
```
Generates summaries for sample conversations and compares to human references.

### 5. Deployment

**FastAPI Service:**
```bash
python src/api/summarization_api.py
```

**API Endpoints:**
- `POST /summarize` - Generate summary for a conversation
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

**Example API request:**
```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": "Agent: Hello! Customer: My laptop is not working...",
    "max_length": 128,
    "num_beams": 4
  }'
```

**Streamlit Dashboard:**
```bash
streamlit run dashboards/summarization_app.py
```
Interactive UI with example conversations and real-time summarization.

---

## 💡 Key Design Decisions

### 1. **Smart Preprocessing Strategy**
Instead of removing file references and URLs entirely (losing context), the system replaces them with descriptive tokens:
- `image.jpg` → `[photo]`
- `https://example.com` → `[link]`
- `document.pdf` → `[file]`

**Rationale:** Preserves conversational context for better summarization quality.

### 2. **Stratified Data Splitting**
Train/val/test splits maintain similar distributions of conversation lengths.

**Rationale:** Prevents bias where the model only sees short conversations in training and fails on long ones in testing.

### 3. **Prefect Over Airflow**
Switched from Airflow to Prefect during development.

**Rationale:** Better compatibility with HuggingFace libraries, faster setup, more Pythonic API.

### 4. **Hybrid Identifier Extraction**
Post-processing extracts order numbers, account IDs using regex and prepends to AI summary.

**Rationale:** SAMSum dataset lacks business identifiers, so hybrid approach ensures critical IDs aren't lost.

### 5. **BART Over T5/Pegasus**
Selected BART-base as the primary model.

**Rationale:** Good balance of performance (competitive ROUGE scores) and efficiency (139M parameters, reasonable inference time).

---

## 🔬 Technical Deep Dive

### Dataset: SAMSum
- **Source:** [HuggingFace - knkarthick/samsum](https://huggingface.co/datasets/knkarthick/samsum)
- **Size:** 14,731 messenger-style conversations with human-written summaries
- **Domain:** General conversations (not customer service specific)
- **Note:** Future work includes fine-tuning on domain-specific CS data

### Model Architecture: BART
- **Base Model:** `facebook/bart-base`
- **Type:** Sequence-to-sequence transformer (encoder-decoder)
- **Parameters:** 139,420,416
- **Max Input:** 512 tokens
- **Max Output:** 128 tokens

### Training Configuration
```yaml
learning_rate: 0.00005
batch_size: 8
num_epochs: 3
warmup_steps: 500
optimizer: AdamW
weight_decay: 0.01
```

### Evaluation Metrics

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
- **ROUGE-1:** Unigram (word) overlap
- **ROUGE-2:** Bigram (2-word phrase) overlap  
- **ROUGE-L:** Longest common subsequence

**Why ROUGE?**
Standard metric for summarization tasks, measures overlap between generated and reference summaries.

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/
```

### Test Individual Components
```bash
# Test database connection
python scripts/test_database.py

# Test dataset loader
python scripts/test_dataset_loader.py

# Test preprocessing
python scripts/test_preprocessor.py

# Test model inference
python scripts/test_model_inference.py
```

---

## 📈 Future Enhancements

### Short-term (Next 2-4 weeks)
- [ ] Add comprehensive unit tests (pytest)
- [ ] Implement response caching in API
- [ ] Add batch processing endpoint
- [ ] Create Docker container for deployment
- [ ] Set up GitHub Actions CI/CD

### Medium-term (1-3 months)
- [ ] Fine-tune on customer service dataset (MultiWOZ)
- [ ] Experiment with T5 and Pegasus models
- [ ] Add sentiment analysis to summaries
- [ ] Implement priority/urgency detection
- [ ] A/B testing framework for model comparison

### Long-term (3-6 months)
- [ ] Multi-language support
- [ ] Real-time streaming summarization
- [ ] Integration with ticketing systems (Zendesk, Freshdesk)
- [ ] Feedback loop for continuous improvement
- [ ] Production monitoring and alerting

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | Prefect | Pipeline workflow management |
| **Data Storage** | SQLite | Conversation data persistence |
| **Data Versioning** | DVC | Dataset version control |
| **Data Validation** | Great Expectations | Quality checks |
| **ML Framework** | PyTorch + Transformers | Model training |
| **Experiment Tracking** | MLflow | Training metrics logging |
| **API** | FastAPI | REST API service |
| **Frontend** | Streamlit | Interactive demo |
| **Configuration** | YAML | Centralized config management |
| **Logging** | Python logging | Debugging and monitoring |

---

## 📚 Documentation

### Configuration Files

**`configs/dataset_config.yaml`**
- Dataset name and source
- Train/val/test split ratios
- Cache directory settings

**`configs/preprocessing_config.yaml`**
- Token replacement rules ([photo], [link], etc.)
- Text cleaning parameters
- Case normalization settings

**`configs/model_config.yaml`**
- Model architecture selection
- Training hyperparameters
- Generation parameters (beam search, length penalty)
- MLflow tracking settings

### Logging
All components use centralized logging:
- Console output for immediate feedback
- File logging in `logs/` directory
- Timestamps and log levels for debugging

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


---

## 👤 Author

**Manjusha**

This project was built as a comprehensive demonstration of MLOps skills, covering:
- Data engineering and ETL pipelines
- Exploratory data analysis
- ML model training and evaluation
- API deployment and serving
- Interactive UI development
- Production-ready system design

---

## 🙏 Acknowledgments

- **Dataset:** SAMSum dataset from HuggingFace
- **Model:** BART from Facebook AI Research
- **Frameworks:** Transformers, FastAPI, Streamlit, Prefect

---

## 📧 Contact

For questions or collaboration opportunities, please reach out via [motamarry.m@northeastern.edu] or open an issue.

---
