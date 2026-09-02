ServiceSense AI

Intelligent Customer Complaint Classification & Resolution Suggestion System

Research-Based Confidence-Aware Hybrid ML + RAG Framework

ServiceSense AI is a research-oriented customer complaint analysis system that combines Machine Learning (ML) and Retrieval-Augmented Generation (RAG) for complaint intent classification and resolution support.

The system uses Logistic Regression for complaint intent classification and TF-IDF with FAISS for retrieving similar historical complaints and their associated resolutions. A validation-selected similarity threshold of 0.85 is used by the proposed hybrid framework to decide between RAG Retrieval and ML Classification. Unsupported or low-confidence complaints can be routed to Manual Review.

Research Problem

Traditional ML-based complaint systems can classify customer intents accurately, but they do not directly retrieve relevant historical resolutions. RAG-based retrieval can provide resolution support, but low-similarity retrieval may produce unreliable results.

ServiceSense AI addresses this problem using a confidence-aware hybrid decision framework.

Research Objectives

Develop a customer complaint intent classifier using Machine Learning.

Integrate RAG-based retrieval for similar historical complaints and resolutions.

Develop a Confidence-Aware Hybrid ML + RAG framework.

Select the RAG similarity threshold using validation data.

Compare Logistic Regression, Random Forest, Training-only RAG, and Hybrid ML + RAG.

Route unsupported or low-confidence complaints to Manual Review.

Evaluate models using Accuracy, Precision, Recall, and F1 Score.

Proposed Method

The research pipeline follows:

Customer Complaint → TF-IDF → ML Classification + FAISS RAG Retrieval → Similarity Evaluation → Confidence-Aware Hybrid Decision → Final Intent → Resolution Suggestion

Decision logic:

Similarity ≥ 0.85 → RAG Retrieval

Similarity < 0.85 → ML Classifier

Unsupported / very low-confidence complaint → Manual Review

The threshold of 0.85 was selected using validation data before final test evaluation.

Final Research Results

Method

Accuracy

F1 Score

Logistic Regression

97.49%

97.50%

Random Forest

97.28%

97.29%

Training-only RAG

95.26%

95.26%

Proposed Hybrid ML + RAG

97.45%

97.45%

Logistic Regression achieved the highest standalone classification accuracy. The proposed Hybrid framework provides comparable classification performance while adding retrieval-based resolution support and confidence-aware routing.

Statistical Analysis

McNemar's test was used to compare Logistic Regression and the Hybrid ML + RAG approach.

Logistic Regression correct / Hybrid wrong: 84

Logistic Regression wrong / Hybrid correct: 47

P-value: 0.001558

The paired prediction difference is statistically significant. The result does not show that the Hybrid classifier is superior in classification accuracy; its research value is in combining classification with retrieval-based resolution support and confidence-aware decision making.

Research Contribution

The main contribution of ServiceSense AI is a Confidence-Aware Hybrid ML + RAG framework that:

Combines ML intent classification with retrieval-based complaint analysis.

Uses a validation-selected similarity threshold for dynamic decision making.

Retrieves relevant historical resolution information.

Routes unsupported complaints to Manual Review instead of forcing an unreliable automated prediction.

Integrates classification, retrieval, confidence evaluation, and resolution support into one pipeline.

Research Demo Scenarios

The application demonstrates three decision paths:

RAG Retrieval — used when retrieval similarity is at or above the selected threshold.

ML Classifier — used when a complaint is supported but RAG similarity is below the threshold.

Manual Review — used for unsupported or very low-confidence complaints.

Technologies Used

Python

Jupyter Notebook

Flask

Scikit-learn

Logistic Regression

Random Forest

TF-IDF

FAISS

Pandas

NumPy

MLflow

Git / GitHub

Main Research Notebook

ServiceSense_AI_Research.ipynb

The notebook contains the complete research workflow, including:

Problem statement and research gap

Research objectives

Dataset preparation

Baseline ML experiments

Training-only RAG evaluation

Validation-based threshold selection

Final Hybrid ML + RAG evaluation

Model comparison

McNemar statistical significance test

Intent confusion analysis

Research findings

Novelty and contribution

Limitations and future work

Final research conclusion

Important Note

The Training-only RAG evaluation uses only training data to construct the retrieval knowledge base. This avoids test-data leakage and provides a fair research evaluation.

Conclusion

The proposed Hybrid ML + RAG framework achieved 97.45% accuracy and 97.45% F1 score. Although Logistic Regression achieved slightly higher classification accuracy at 97.49%, the Hybrid framework adds retrieval-based resolution support, similarity-based validation, confidence-aware decision making, and Manual Review handling for unsupported complaints.

Therefore, the primary research contribution is improved decision support and reliability through the integration of ML classification and RAG retrieval rather than a claim of higher classification accuracy.