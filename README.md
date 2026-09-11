# ServiceSense AI

## Intelligent Customer Complaint Classification & Resolution Suggestion System

ServiceSense AI is a research-based customer-support system that combines Machine Learning and Retrieval-Augmented Generation (RAG) for complaint classification and resolution support.

The application demonstrates the proposed framework using a **Flipkart e-commerce customer-support use case**.

> **Academic Note:** Flipkart is used only as an e-commerce demonstration use case. This project does not use internal Flipkart data and is not affiliated with Flipkart.

---

## Project Objective

The main objectives of ServiceSense AI are:

- Classify customer complaints automatically.
- Identify the intent of a customer complaint.
- Retrieve relevant complaint-resolution knowledge.
- Suggest an appropriate resolution.
- Compare traditional Machine Learning models with RAG.
- Develop a confidence-aware Hybrid ML + RAG framework.
- Send unsupported or uncertain complaints for manual review.
- Demonstrate the research framework through a Flask web application.

---

## Dataset

The project uses a publicly available customer-support dataset for academic research.

Dataset file:

```text
data/ServiceSense.csv
```

The dataset contains approximately **26,872 customer-support records**.

Important fields used in the project:

- `instruction` — customer complaint/request
- `category` — complaint category
- `intent` — customer intent
- `response` — corresponding support response

The research dataset contains **11 categories and 27 intents**.

---

## Research Methodology

The research compares four approaches:

1. Logistic Regression
2. Random Forest
3. Training-only RAG
4. Confidence-Aware Hybrid ML + RAG

TF-IDF is used for text representation.

FAISS is used for similarity-based retrieval in the RAG component.

The Hybrid framework combines ML classification and RAG retrieval using a validation-selected confidence threshold.

---

## System Architecture

```text
Customer Complaint
        |
        v
Text Preprocessing
        |
        v
      TF-IDF
        |
        +-------------------------+
        |                         |
        v                         v
ML Classification           RAG Retrieval
(Logistic Regression)       (TF-IDF + FAISS)
        |                         |
        v                         v
ML Intent + Confidence      Similar Complaint +
                            Similarity + Resolution
        |                         |
        +------------+------------+
                     |
                     v
        Confidence-Aware Hybrid Decision
              Threshold = 0.85
                     |
             +-------+-------+
             |               |
             v               v
      Accepted Result    Low Confidence /
                         Unsupported
             |               |
             v               v
 Category + Intent      Manual Review
 + Resolution
 + Priority
 + Recommended Action
```

---

## Experimental Results

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 97.49% | 97.50% |
| Random Forest | 97.28% | 97.29% |
| Training-only RAG | 95.26% | 95.26% |
| Hybrid ML + RAG | 97.45% | 97.45% |

### Threshold Optimization

The Hybrid decision threshold was selected using validation experiments.

```text
Best Threshold = 0.85
Validation F1 Score = 97.47%
```

### Final Unbiased Hybrid Test Result

```text
Accuracy  = 97.45%
Precision = 97.50%
Recall    = 97.45%
F1 Score  = 97.45%
```

The Logistic Regression model achieved slightly higher pure classification performance than the Hybrid model.

The contribution of the Hybrid approach is not simply higher classification accuracy. It provides additional retrieval-based resolution support, confidence-aware decision making, and manual-review handling for unsupported complaints.

---

## Flipkart E-Commerce Demonstration

The Flask prototype demonstrates how ServiceSense AI can be applied to an e-commerce customer-support environment such as Flipkart.

Example complaint types include:

- Order-related complaints
- Delivery and tracking issues
- Payment problems
- Refund requests
- Order cancellation
- Account and password issues
- Damaged or incorrect product return requests

The demo also contains domain safety handling for clear delivery and return-related complaints.

These demonstration rules are application-level safety features and are **not used to calculate the reported research evaluation metrics**.

---

## Technology Stack

### Programming

```text
Python
```

### Machine Learning

```text
Scikit-learn
Logistic Regression
Random Forest
```

### NLP and RAG

```text
TF-IDF
FAISS
Retrieval-Augmented Generation (RAG)
```

### Web Application

```text
Flask
HTML
CSS
```

### MLOps and Version Control

```text
MLflow
Git
GitHub
```

### Deployment

```text
Docker
```

### Development Environment

```text
VS Code
Jupyter Notebook
```

---

## Project Structure

```text
ServiceSense_AI/
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
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── ServiceSense_AI.ipynb
├── ServiceSense_AI_Research.ipynb
└── README.md
```

---

## Run Locally

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# Docker Deployment

ServiceSense AI is containerized and deployed using **Docker**.

## 1. Check Docker

```bash
docker --version
```

## 2. Build Docker Image

From the ServiceSense AI project directory:

```bash
docker build -t servicesense-ai .
```

## 3. Run Docker Container

```bash
docker run --name servicesense-container -p 5000:5000 servicesense-ai
```

Open the application:

```text
http://127.0.0.1:5000
```

## 4. Check Running Container

```bash
docker ps
```

The container should appear as:

```text
servicesense-container
```

## 5. Stop Container

```bash
docker stop servicesense-container
```

## 6. Start Container Again

```bash
docker start servicesense-container
```

## 7. Automatic Restart

Configure the ServiceSense AI container to restart automatically when Docker restarts:

```bash
docker update --restart unless-stopped servicesense-container
```

Verify the restart policy:

```bash
docker inspect -f "{{.HostConfig.RestartPolicy.Name}}" servicesense-container
```

Expected output:

```text
unless-stopped
```

---

## REST API

ServiceSense AI also provides a REST API.

### Prediction Endpoint

```text
POST /api/predict
```

Example JSON request:

```json
{
  "complaint": "My payment failed but money was deducted from my account."
}
```

The API returns information such as:

```text
Complaint Category
Predicted Intent
Suggested Resolution
Similarity Score
Confidence Level
Priority
Recommended Action
ML Intent
RAG Intent
ML Confidence
RAG Similarity
Decision Source
Validation Threshold
```

---

## Health Check

Health endpoint:

```text
GET /api/health
```

Local URL:

```text
http://127.0.0.1:5000/api/health
```

---

## Research Contribution

The proposed ServiceSense AI framework provides:

- Comparative evaluation of traditional ML and RAG.
- Confidence-aware Hybrid ML + RAG decision making.
- Validation-based threshold selection.
- Retrieval-based customer resolution support.
- Manual review for unsupported complaints.
- Explainable ML and RAG decision information.
- An e-commerce application prototype demonstrating the research framework.

---

## Important Research Note

The reported experimental results were obtained from the controlled research evaluation.

The Docker-based Flipkart demonstration is the **application prototype** used to demonstrate the proposed framework.

Application-level safety rules added to the demo do not replace or modify the original research evaluation results.

---

## Project Type

**Research-Based Project with Application Prototype**

### Project

**ServiceSense AI — Intelligent Customer Complaint Classification & Resolution Suggestion System**

### Demonstration Use Case

**Flipkart E-Commerce Customer Support**

### Deployment

**Docker**