# 🌿 AI-Driven Treatment Advisory System for Pomegranate Disease Diagnosis and Severity Analysis

This repository contains the implementation of a deep learning-powered, end-to-end AI system that classifies pomegranate fruit diseases, estimates severity, and generates severity-specific treatment recommendations using Retrieval-Augmented Generation (RAG) and a domain-specific Large Language Model (LLM).

## 📌 Project Highlights

- **99.35% Classification Accuracy** using DenseNet121 on a curated 5,099-image dataset covering 5 pomegranate disease categories.
- **Severity Estimation** using a novel **Healthy-Based Deviation Scoring (HBDS)** method combining:
  - Grad-CAM++ for lesion localization  
  - Mahalanobis Distance for feature deviation  
  - GMM for severity-level clustering (Low, Medium, High)
- **RAG-based Treatment Generator** using:
  - Qdrant vector database for semantic search  
  - Mistral Small 3.1 LLM for grounded response generation
- **Deployment-Ready Flask App**:
  - Image upload, prediction + severity visualization  
  - Chatbot interface for real-time queries  
  - Downloadable PDF treatment reports

## 🖼 Sample Output

![image](https://github.com/user-attachments/assets/957e501d-7e14-4536-a9d6-60e369fc34cc)
![image](https://github.com/user-attachments/assets/6320071c-2d8e-475b-8cd9-c057e74405ed)
![image](https://github.com/user-attachments/assets/6afa6412-6ac7-4be3-b6a5-83bf51e55789)

📌 Future Work

Add multilingual support for regional Indian languages
Expand to other crops (mango, guava, citrus)
Deploy on cloud / edge (Raspberry Pi, Android)





