import joblib

def test_model_loading():
    model = joblib.load("models/model_a/traditional/hard_voting_ensemble.pkl")
    assert model is not None

def test_vectorizer_loading():
    vectorizer = joblib.load("models/model_a/traditional/tfidf_vectorizer.pkl")
    assert vectorizer is not None