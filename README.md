# 🤖 BullyNet.ai – AI-Powered Cyberbullying Detection Chatbot  

**BullyNet.ai** is an **AI-driven chatbot** built to detect and respond to **cyberbullying, hate speech, and offensive language** using advanced **Natural Language Processing (NLP)** and **Deep Learning** techniques.  
The project focuses on promoting **safer online communication** and demonstrates expertise in **Machine Learning, NLP pipelines, sentiment analysis, and AI ethics**.  

---

## Key Highlights  

- 🔍 **Real-time Bully Detection:** Instantly identifies toxic or abusive text with high accuracy.  
- 💬 **Interactive Chatbot:** Engages users and provides educational or awareness-based responses.  
- 🧠 **Deep Learning Models:** Trained using **LSTM** and **CNN** architectures for multi-class classification.  
- 📊 **Data-Driven Insights:** Analyzes emotional tone, sentiment score, and bullying intensity.  
- 🌍 **AI for Social Good:** Supports mental well-being by promoting responsible communication online.  

---

## Tech Stack  

| **Category** | **Tools / Frameworks** |
|---------------|-------------------------|
| Programming Language | Python |
| Deep Learning | TensorFlow, PyTorch |
| NLP Libraries | NLTK, spaCy |
| ML Algorithms | LSTM, CNN |
| Data Visualization | Matplotlib, Seaborn |
| Model Evaluation | Scikit-learn, Confusion Matrix, ROC Curve |
| Deployment | Google Colab, GitHub |
| Dataset | Kaggle – *Hate Speech and Offensive Language Dataset* (by Andrii Samoshyn) |

---

## Dataset Details  

- **Dataset Name:** Hate Speech and Offensive Language Dataset  
- **Size:** ~25,000 labeled tweets and comments  
- **Classes:**  
  - `0` → Normal (Non-offensive)  
  - `1` → Offensive Language  
  - `2` → Hate Speech / Bullying  
-  **Preprocessing Performed:**  
  - Text cleaning (removal of links, hashtags, emojis)  
  - Tokenization & Lemmatization  
  - Stopword removal using **NLTK**  
  - Vectorization with **TF-IDF** and **Word2Vec**  

---

## Workflow  

1. **Data Collection & Cleaning** – Load dataset and preprocess text (remove noise, lowercase, lemmatize).  
2. **Feature Engineering** – Convert text to numerical form using **TF-IDF / Embeddings**.  
3. **Model Training** – Train multiple models (LSTM, CNN) to classify text into bullying categories.  
4. **Model Evaluation** – Measure accuracy, precision, recall, and F1-score.  
5. **Chatbot Integration** – Connect model with a conversational chatbot for live text moderation.  
6. **Deployment** – Deployed and tested via **Google Colab** for prototype demonstration.  

---
##  Model Architectures  

### 1️. Logistic Regression (Baseline)
- Trained using TF-IDF features.  
- Accuracy: **0.87**

### 2️. LSTM Model
- Embedding Layer (128 units)  
- LSTM Layer (128 units, dropout 0.2)  
- Dense output with sigmoid activation  
- Accuracy: **0.91**

### 3️. CNN Model
- Embedding Layer  
- 1D Convolution Layer (128 filters, kernel=5)  
- MaxPooling + Dense + Dropout  
- Accuracy: **0.92**

### 4️. Hybrid CNN + LSTM Model (Best)
- Embedding Layer → Conv1D → MaxPooling → LSTM → Dense  
- Accuracy: **0.94**, F1-score: **0.93**

----

##  Results Summary  

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|-----------|---------|-----------|
| Logistic Regression | 0.87 | 0.86 | 0.85 | 0.85 |
| LSTM | 0.91 | 0.90 | 0.89 | 0.89 |
| CNN | 0.92 | 0.91 | 0.91 | 0.91 |
| **CNN + LSTM (Hybrid)** | **0.94** | **0.93** | **0.92** | **0.93** |

-----

## Achievements  

- Achieved **~90% accuracy** on test data using tuned LSTM model.  
- Created **modular NLP pipeline** for preprocessing and classification.  
- Integrated deep learning model with a chatbot interface.  
- Demonstrated practical application of **AI Ethics** and **Responsible AI** principles.  

---

## Future Enhancements  

- Integration with **BERT / Transformer-based architectures** for improved context understanding.  
- Support for **multi-language detection** (English, Hindi, and regional Indian languages).  
- Development of **real-time moderation dashboard** for communities, schools, and forums.  
- Adding **emotion recognition** to handle user distress and provide mental health resources.  
