from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
import os
import sys

# ---------------- APP CONFIG ----------------
app = Flask(__name__)
app.secret_key = "secret123"

# Reduce CPU load
torch.set_num_threads(2)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "retrieval_model")

FAISS_PATH = os.path.join(MODEL_DIR, "faiss.index")
CSV_PATH = os.path.join(MODEL_DIR, "train_data.csv")

# ---------------- LOAD MODEL & DATA ----------------
try:
    print("🔄 Loading sentence embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("🔄 Loading FAISS index...")
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"FAISS file not found: {FAISS_PATH}")
    index = faiss.read_index(FAISS_PATH)

    print("🔄 Loading training data...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    train_df = pd.read_csv(CSV_PATH)

    train_questions = train_df["questions"].tolist()
    train_answers = train_df["answers"].tolist()

    print("✅ System ready")

except Exception as e:
    print("❌ ERROR DURING STARTUP")
    print(e)
    sys.exit(1)

# ---------------- RETRIEVAL FUNCTION ----------------
def retrieve_answer(question):
    embedding = embedder.encode(
        [question.lower()],
        normalize_embeddings=True
    )

    scores, indices = index.search(embedding, 1)

    idx = int(indices[0][0])
    score = float(scores[0][0])

    return {
        "retrieved_question": train_questions[idx],
        "answer": train_answers[idx],
        "confidence": round(score * 100, 2)
    }

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["user"] = request.form.get("username")
        return redirect(url_for("predict"))
    return render_template("login.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        question = request.form.get("question")

        result = retrieve_answer(question)

        return render_template(
            "result.html",
            question=question,
            predicted_answer=result["answer"],
            actual_answer=result["answer"],   # expert answer
            confidence=result["confidence"]
        )

    return render_template("predict.html")

@app.route("/chart")
def chart():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("chart.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True, host="127.0.0.1", port=5000)
