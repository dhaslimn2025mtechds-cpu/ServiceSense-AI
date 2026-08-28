# ServiceSense AI

ServiceSense AI is an intelligent customer complaint classification and resolution suggestion system.

## Features

- Customer complaint classification
- Intent prediction using Machine Learning
- TF-IDF text vectorization
- FAISS similarity search
- Resolution suggestion from similar historical complaints
- Similarity score
- Flask web application
- REST API endpoint
- MLflow experiment tracking

## Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF
- FAISS
- Flask
- MLflow
- Joblib
- Postman

## Project Workflow

Customer Complaint
→ Text Preprocessing
→ TF-IDF Vectorization
→ Intent Classification
→ FAISS Similarity Search
→ Similar Complaint Retrieval
→ Resolution Suggestion
→ Flask Web Application / REST API

## Project Structure

```text
ServiceSense_AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── ServiceSense.csv
│
├── models/
│   ├── intent_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── rag_tfidf_vectorizer.pkl
│   └── complaints_faiss.index
│
├── templates/
│   └── index.html
│
├── static/
│
└── ServiceSense_AI.ipynb