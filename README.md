# 📘 Intelligent Reading Comprehension & Quiz Generation System

## 🚀 Project Overview

This project implements an AI-powered system for reading comprehension and quiz generation using the RACE dataset.

The system consists of two main components:

* **Model A (Answer Verifier):** Predicts the correct answer from multiple-choice options.
* **Model B (Distractor & Hint Generator):** Generates plausible incorrect options and helpful hints.

A **Streamlit-based UI** is provided for interactive usage.

---

## 🧠 Features

### 🔹 Model A

* TF-IDF feature extraction
* Logistic Regression & Linear SVM
* Answer verification (correct vs incorrect)
* End-to-end MCQ prediction

### 🔹 Model B

* Distractor generation using cosine similarity
* Hint generation (3 levels: general → specific)
* Sentence ranking from passage

### 🔹 UI (Streamlit)

* Article input / random sample
* Quiz interface (MCQ)
* Hint panel
* Developer dashboard (metrics + logs)

---

## 📊 Dataset

* Dataset: **RACE (Reading Comprehension Dataset)**
* ~62k training samples used
* Each sample includes:

  * Article
  * Question
  * 4 options (A, B, C, D)
  * Correct answer

---

## 📈 Model Performance

| Model               | Accuracy | Macro F1 |
| ------------------- | -------- | -------- |
| Logistic Regression | ~0.507   | ~0.474   |
| Linear SVM          | ~0.501   | ~0.470   |
| K-Means             | ~0.75*   | ~0.43    |

*Accuracy inflated due to class imbalance

---

## 🛠️ Tech Stack

* Python
* pandas, numpy
* scikit-learn (TF-IDF, ML models)
* Streamlit (UI)
* joblib (model persistence)

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
python -m streamlit run ui/app.py
```

### 3. Open in browser

```
http://localhost:8501
```

---

## 📁 Project Structure

```
race_rc_project/
├── data/
├── models/
├── src/
├── ui/
│   └── app.py
├── README.md
├── requirements.txt
```

---

## ⚠️ Limitations

* Uses classical ML (no deep learning)
* TF-IDF lacks deep semantic understanding
* Distractors may not always be perfectly natural
* Clustering performance is limited in high-dimensional text space

---

## 🔮 Future Improvements

* Use transformer models (BERT, RoBERTa)
* Improve distractor generation with NLP techniques
* Add automatic question generation
* Enhance UI with analytics and scoring

---

## ⚖️ Ethical Considerations

* AI-generated content may contain errors
* Human validation is recommended
* System should not be used for critical assessments without review

---

## 👨‍💻 Author

Hassaan Qadir and Huzaifa Sajjad
