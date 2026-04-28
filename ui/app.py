
import streamlit as st
import pandas as pd
import joblib
import time
import re
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Basic Helpers
# -----------------------------

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', str(text))
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def get_similarity_scores(vectorizer, question, sentences):
    texts = [question] + sentences
    tfidf = vectorizer.transform(texts)
    return cosine_similarity(tfidf[0], tfidf[1:])[0]

def generate_distractors(article, question, correct_answer, vectorizer, top_k=10):
    sentences = split_sentences(article)

    if len(sentences) == 0:
        return ["No distractor available"] * 3

    scores = get_similarity_scores(vectorizer, question, sentences)
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

    candidates = []

    for sent, score in ranked[:top_k]:
        words = sent.split()

        for i in range(len(words)):
            phrase = " ".join(words[i:i+5])

            if len(phrase) > 10 and phrase.lower() not in correct_answer.lower():
                candidates.append(phrase)

    candidates = list(dict.fromkeys(candidates))

    if len(candidates) >= 3:
        return candidates[:3]

    return candidates + ["No distractor available"] * (3 - len(candidates))

def generate_hints(article, question, vectorizer):
    sentences = split_sentences(article)

    if len(sentences) == 0:
        return ["No hint available"] * 3

    scores = get_similarity_scores(vectorizer, question, sentences)
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1])

    hint1 = ranked[0][0]
    hint2 = ranked[len(ranked)//2][0]
    hint3 = ranked[-1][0]

    return [hint1, hint2, hint3]

# def predict_answer(model, vectorizer, article, question, options):
    best_label = None
    best_score = -999

    for label, option_text in options.items():
        combined = article + " " + article + " " + question + " " + option_text
        X = vectorizer.transform([combined])

        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X)[0][1]
        else:
            score = model.decision_function(X)[0]

        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score
def predict_answer(model_bundle, vectorizer, article, question, options):
    models = model_bundle["models"]

    best_label = None
    best_score = -999

    for label, option_text in options.items():
        combined = article + " " + article + " " + question + " " + option_text
        X = vectorizer.transform([combined])

        votes = []

        for model in models:
            pred = model.predict(X)[0]
            votes.append(pred)

        final_vote = max(set(votes), key=votes.count)

        score = votes.count(final_vote)

        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score
# -----------------------------
# Load Assets
# -----------------------------

# @st.cache_resource
# def load_assets():
#     # vectorizer = joblib.load("race_rc_project/models/model_a/traditional/tfidf_vectorizer.pkl")
#     # model = joblib.load("race_rc_project/models/model_a/traditional/best_model.pkl")
#     vectorizer = joblib.load("models/model_a/traditional/tfidf_vectorizer.pkl")
#     # model = joblib.load("models/model_a/traditional/best_model.pkl")
#     ensemble = joblib.load("models/model_a/traditional/hard_voting_ensemble.pkl")
#     return vectorizer, model

@st.cache_resource
def load_assets():
    vectorizer = joblib.load("models/model_a/traditional/tfidf_vectorizer.pkl")
    model = joblib.load("models/model_a/traditional/hard_voting_ensemble.pkl")
    return vectorizer, model

@st.cache_data
def load_data():
    # return pd.read_csv("race_rc_project/data/raw/test.csv")
    return pd.read_csv("data/raw/test.csv")

vectorizer, model = load_assets()
test_df = load_data()

# -----------------------------
# UI Config
# -----------------------------

st.set_page_config(
    page_title="AI Reading Comprehension Quiz Generator",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Intelligent Reading Comprehension & Quiz Generator")
st.caption("Classical ML system using TF-IDF, Logistic Regression/SVM, distractor generation, and hint extraction.")

if "logs" not in st.session_state:
    st.session_state.logs = []

if "sample" not in st.session_state:
    st.session_state.sample = None

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose Screen",
    [
        "1. Article Input",
        "2. Quiz View",
        "3. Hint Panel",
        "4. Developer Dashboard"
    ]
)

# -----------------------------
# Screen 1
# -----------------------------

if page == "1. Article Input":
    st.header("Screen 1 — Article Input")

    col1, col2 = st.columns([2, 1])

    with col1:
        article_input = st.text_area(
            "Paste a reading passage",
            height=250,
            placeholder="Paste article text here..."
        )

    with col2:
        st.subheader("Quick Test")
        if st.button("Load Random RACE Sample"):
            st.session_state.sample = test_df.sample(1).iloc[0].to_dict()
            st.success("Random sample loaded.")

        if st.session_state.sample:
            st.write("Loaded question:")
            st.info(st.session_state.sample["question"])

    if st.button("Submit Article"):
        if not article_input.strip() and st.session_state.sample is None:
            st.error("Please paste an article or load a random RACE sample.")
        else:
            st.success("Article accepted. Go to Quiz View.")

# -----------------------------
# Screen 2
# -----------------------------

elif page == "2. Quiz View":
    st.header("Screen 2 — Question & Answer Quiz View")

    if st.session_state.sample is None:
        st.warning("Please load a random RACE sample from Screen 1 first.")
    else:
        sample = st.session_state.sample

        article = sample["article"]
        question = sample["question"]
        correct_label = sample["answer"]
        correct_answer = sample[correct_label]

        options = {
            "A": sample["A"],
            "B": sample["B"],
            "C": sample["C"],
            "D": sample["D"]
        }

        st.subheader("Article")
        with st.expander("Read Passage"):
            st.write(article)

        st.subheader("Question")
        st.write(question)

        user_choice = st.radio(
            "Choose your answer:",
            list(options.keys()),
            format_func=lambda x: f"{x}. {options[x]}"
        )

        if st.button("Check Answer"):
            start = time.time()

            predicted_label, score = predict_answer(
                model, vectorizer, article, question, options
            )

            latency = time.time() - start
            is_user_correct = user_choice == correct_label
            model_correct = predicted_label == correct_label

            if is_user_correct:
                st.success(f"Correct! The answer is {correct_label}.")
            else:
                st.error(f"Incorrect. Correct answer: {correct_label}. {correct_answer}")

            st.info(f"Model predicted: {predicted_label} | Latency: {latency:.3f}s")

            st.session_state.logs.append({
                "question": question,
                "correct_answer": correct_label,
                "user_answer": user_choice,
                "model_prediction": predicted_label,
                "user_correct": is_user_correct,
                "model_correct": model_correct,
                "latency": latency
            })

# -----------------------------
# Screen 3
# -----------------------------

elif page == "3. Hint Panel":
    st.header("Screen 3 — Hint Panel")

    if st.session_state.sample is None:
        st.warning("Please load a random RACE sample from Screen 1 first.")
    else:
        sample = st.session_state.sample

        article = sample["article"]
        question = sample["question"]
        correct_label = sample["answer"]
        correct_answer = sample[correct_label]

        hints = generate_hints(article, question, vectorizer)

        st.subheader("Question")
        st.write(question)

        with st.expander("Hint 1 — General"):
            st.write(hints[0])

        with st.expander("Hint 2 — More Specific"):
            st.write(hints[1])

        with st.expander("Hint 3 — Near Explicit"):
            st.write(hints[2])

        if st.button("Reveal Answer"):
            st.success(f"Correct answer: {correct_label}. {correct_answer}")

# -----------------------------
# Screen 4
# -----------------------------

elif page == "4. Developer Dashboard":
    st.header("Screen 4 — Developer / Analytics Dashboard")

    st.subheader("Model A Performance")
    st.write("Validation Results:")
    st.dataframe(pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Accuracy": 0.507,
            "Precision": 0.256,
            "Recall": 0.510,
            "Macro F1": 0.474
        },
        {
            "Model": "Linear SVM",
            "Accuracy": 0.501,
            "Precision": 0.255,
            "Recall": 0.518,
            "Macro F1": 0.470
        },
        {
            "Model": "K-Means",
            "Accuracy": 0.750,
            "Precision": "N/A",
            "Recall": "N/A",
            "Macro F1": 0.429
        }
    ]))

    st.subheader("Session Logs")

    if len(st.session_state.logs) == 0:
        st.info("No quiz attempts logged yet.")
    else:
        logs_df = pd.DataFrame(st.session_state.logs)
        st.dataframe(logs_df)

        csv = logs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Session Logs as CSV",
            csv,
            "session_logs.csv",
            "text/csv"
        )

    st.subheader("Model B Evaluation")
    st.write("""
    Distractor validity accuracy: 1.0  
    Evaluation checks whether generated distractors are different from the correct answer.
    Further manual evaluation can rate plausibility, diversity, and grammatical consistency.
    """)

    st.warning("AI-generated questions, answers, hints, and distractors may contain errors. Human review is recommended.")
