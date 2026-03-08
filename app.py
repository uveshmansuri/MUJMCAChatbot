import os
import json
import numpy as np
import streamlit as st
import google.genai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ================================
# CONFIG
# ================================
EMBED_MODEL = "BAAI/bge-m3"
model = SentenceTransformer(EMBED_MODEL)

DEFAULT_API_KEY = st.secrets["GEMINI_API_KEY"]


# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="MUJ MCA Sem1 Q&A", layout="wide")

st.title("📄 MUJ MCA Semester 1 Question Answering (Subject-Wise)")

st.markdown(
"""
Ask questions from your processed document databases.
The AI will answer **only using the document context**.
"""
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Settings")

db_files = [f for f in os.listdir() if f.endswith("_db.json")]

if not db_files:
    st.sidebar.error("No database files found!")
    st.stop()

# Create display names (remove _db.json)
subject_map = {f.replace("_db.json", ""): f for f in db_files}

selected_subject = st.sidebar.selectbox(
    "Select Subject",
    list(subject_map.keys())
)

# Get actual file name
selected_db = subject_map[selected_subject]


# API key input
user_api_key = st.sidebar.text_input(
    "Gemini API Key (optional)",
    type="password"
)

api_key = user_api_key if user_api_key else DEFAULT_API_KEY


# ================================
# LOAD DB
# ================================
@st.cache_resource
def load_embeddings(db_file):
    with open(db_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["chunks"], np.array(data["embeddings"])


chunks, embeddings = load_embeddings(selected_db)


# ================================
# PROMPT
# ================================
def get_prompt(context_text, user_question, word_limit):
    return f"""
You are an intelligent document assistant. Answer the user's question using ONLY the provided document context.
---------------------
DOCUMENT CONTEXT
{context_text}
---------------------
USER QUESTION
{user_question}
---------------------
INSTRUCTIONS:
1. Use the provided context as the main source of information.
2. Base your answer strictly on the context. Do not add new facts that are not supported by it.
3. Match the document’s style: Follow the same tone, terminology, and sentence style used in the context, Prefer similar wording where appropriate to preserve the document’s meaning.
4. Write the answer in simple, clear English that is easy to understand.
5. The answer MUST NOT exceed **{word_limit} words**.
6. Keep the answer concise and avoid unnecessary words.
7. Formatting rules:
   - If the question asks for steps, reasons, advantages, differences, or multiple items → respond in clean bullet points.
   - If the question asks for a definition or explanation → respond in 2–4 short sentences.
8. If the answer is not present in the context or the question is unrelated to the document topic, respond exactly with: "I'm not sure based on the provided document."
9. Do not repeat the question.
10. Do not mention the context or these instructions in the answer.
"""


# ================================
# SEARCH
# ================================
def search_chunks(question, top_k=15, threshold=0.5):

    q_emb = model.encode([question], normalize_embeddings=True)
    sims = cosine_similarity(q_emb, embeddings)[0]

    above_threshold = [(idx, sim) for idx, sim in enumerate(sims) if sim >= threshold]

    if not above_threshold:
        top_idx = int(np.argmax(sims))
        return [chunks[top_idx]]

    above_threshold.sort(key=lambda x: x[1], reverse=True)

    return [chunks[idx] for idx, _ in above_threshold[:top_k]]


# ================================
# GEMINI
# ================================
def generate_answer(question, context_chunks, word_limit):

    context_text = "\n\n".join(context_chunks)
    prompt = get_prompt(context_text, question, word_limit)

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# ================================
# QUESTION INPUT
# ================================
question = st.text_input("Ask your question")

# Word range inputs
st.sidebar.subheader("Answer Length Range")

min_words = st.sidebar.number_input(
    "Minimum words",
    min_value=80,
    max_value=2000,
    step=10
)

max_words = st.sidebar.number_input(
    "Maximum words",
    min_value=100,
    max_value=2000,
    step=10
)

if st.button("Get Answer"):

    if not question.strip():
        st.warning("Please enter a question")
        st.stop()

    if not api_key:
        st.error("Please provide a Gemini API key")
        st.stop()

    with st.spinner("🔎 Searching in syllabus and generating answer..."):

        context = search_chunks(question)

        try:
            word_limit = [min_words, max_words]
            answer = generate_answer(question, context, word_limit)

            st.success("Answer Found")

            st.markdown("### Your Answer")
            st.write(answer)

        except Exception as e:
            st.error(f"Error: {e}")