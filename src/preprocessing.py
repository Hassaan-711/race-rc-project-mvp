
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from scipy.sparse import save_npz


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def jaccard_similarity(text1, text2):
    set1 = set(clean_text(text1).split())
    set2 = set(clean_text(text2).split())

    if len(set1) == 0 or len(set2) == 0:
        return 0

    return len(set1.intersection(set2)) / len(set1.union(set2))


def build_verification_dataset(df):
    rows = []

    for _, row in df.iterrows():
        article = str(row["article"])
        question = str(row["question"])
        correct_answer = row["answer"]

        for option_label in ["A", "B", "C", "D"]:
            option_text = str(row[option_label])

            rows.append({
                "id": row["id"],
                "article": article,
                "question": question,
                "option_label": option_label,
                "option_text": option_text,
                "combined_text": article + " " + article + " " + question + " " + option_text,
                "is_correct": 1 if option_label == correct_answer else 0
            })

    return pd.DataFrame(rows)


def add_handcrafted_features(df):
    df = df.copy()

    df["article_len"] = df["article"].apply(lambda x: len(str(x).split()))
    df["question_len"] = df["question"].apply(lambda x: len(str(x).split()))
    df["option_len"] = df["option_text"].apply(lambda x: len(str(x).split()))

    df["question_option_overlap"] = df.apply(
        lambda row: jaccard_similarity(row["question"], row["option_text"]),
        axis=1
    )

    df["article_option_overlap"] = df.apply(
        lambda row: jaccard_similarity(row["article"], row["option_text"]),
        axis=1
    )

    df["article_question_overlap"] = df.apply(
        lambda row: jaccard_similarity(row["article"], row["question"]),
        axis=1
    )

    return df


def main():
    train_df = pd.read_csv("race_rc_project/data/raw/train.csv")
    val_df = pd.read_csv("race_rc_project/data/raw/val.csv")
    test_df = pd.read_csv("race_rc_project/data/raw/test.csv")

    train_verif = add_handcrafted_features(build_verification_dataset(train_df))
    val_verif = add_handcrafted_features(build_verification_dataset(val_df))
    test_verif = add_handcrafted_features(build_verification_dataset(test_df))

    os.makedirs("race_rc_project/data/processed", exist_ok=True)
    os.makedirs("race_rc_project/models/model_a/traditional", exist_ok=True)

    train_verif.to_csv("race_rc_project/data/processed/model_a_train.csv", index=False)
    val_verif.to_csv("race_rc_project/data/processed/model_a_val.csv", index=False)
    test_verif.to_csv("race_rc_project/data/processed/model_a_test.csv", index=False)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train = vectorizer.fit_transform(train_verif["combined_text"])
    X_val = vectorizer.transform(val_verif["combined_text"])
    X_test = vectorizer.transform(test_verif["combined_text"])

    save_npz("race_rc_project/data/processed/X_train_tfidf.npz", X_train)
    save_npz("race_rc_project/data/processed/X_val_tfidf.npz", X_val)
    save_npz("race_rc_project/data/processed/X_test_tfidf.npz", X_test)

    train_verif["is_correct"].to_csv("race_rc_project/data/processed/y_train.csv", index=False)
    val_verif["is_correct"].to_csv("race_rc_project/data/processed/y_val.csv", index=False)
    test_verif["is_correct"].to_csv("race_rc_project/data/processed/y_test.csv", index=False)

    joblib.dump(vectorizer, "race_rc_project/models/model_a/traditional/tfidf_vectorizer.pkl")

    print("Preprocessing completed successfully.")
    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)


if __name__ == "__main__":
    main()
