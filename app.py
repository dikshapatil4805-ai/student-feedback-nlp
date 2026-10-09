
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    page_icon="🎓",
    layout="wide"
)

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "helpful", "clear",
    "useful", "love", "liked", "best", "wonderful", "effective",
    "interesting", "friendly", "supportive", "enjoy", "enjoyed",
    "understand", "improved", "impressive", "satisfied", "easy",
    "organized", "engaging", "fantastic", "appreciate"
}

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "slow", "confusing",
    "difficult", "boring", "worst", "hate", "dislike", "unclear",
    "useless", "problem", "problems", "issue", "issues", "late",
    "disappointing", "disappointed", "hard", "noisy", "uncomfortable",
    "ineffective", "waste", "weak", "stressful", "expensive"
}

NEGATIONS = {
    "not", "no", "never", "hardly", "isn't", "wasn't",
    "don't", "didn't", "doesn't", "cannot", "can't"
}


def analyze_sentiment(text):
    if pd.isna(text) or not str(text).strip():
        return "Neutral"

    words = re.findall(r"[a-zA-Z']+", str(text).lower())
    score = 0

    for i, word in enumerate(words):
        if word in POSITIVE_WORDS:
            points = 1
        elif word in NEGATIVE_WORDS:
            points = -1
        else:
            continue

        # Reverse the word's sentiment when preceded by a negation.
        previous_words = words[max(0, i - 3):i]
        if any(w in NEGATIONS for w in previous_words):
            points *= -1

        score += points

    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


st.title("🎓 Student Feedback Analysis")
st.write(
    "Upload student feedback in CSV format to analyze sentiment, "
    "explore visualizations, and download the results."
)

uploaded_file = st.file_uploader("Upload your feedback CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to get started.")

    st.subheader("Example CSV format")
    example = pd.DataFrame({
        "Feedback": [
            "The teachers explain concepts clearly",
            "The laboratory computers are very slow",
            "The course is average"
        ]
    })
    st.dataframe(example, use_container_width=True)
    st.download_button(
        "Download example CSV",
        example.to_csv(index=False).encode("utf-8"),
        "example_feedback.csv",
        "text/csv"
    )
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read this CSV file: {error}")
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("The uploaded CSV has no data.")
    st.stop()

st.subheader("Preview your data")
st.dataframe(df.head(10), use_container_width=True)

feedback_column = st.selectbox(
    "Select the column containing student feedback",
    options=list(df.columns)
)

if st.button("Analyze Feedback", type="primary"):
    result = df.copy()
    result["Sentiment"] = result[feedback_column].apply(analyze_sentiment)

    total = len(result)
    positive = int((result["Sentiment"] == "Positive").sum())
    negative = int((result["Sentiment"] == "Negative").sum())
    neutral = int((result["Sentiment"] == "Neutral").sum())

    st.session_state["analysis_result"] = result

result = st.session_state.get("analysis_result")

if result is not None:
    # Only show the stored results if they belong to the current upload.
    # A new upload resets the result to avoid showing stale data.
    st.subheader("📊 Sentiment Dashboard")

    total = len(result)
    positive = int((result["Sentiment"] == "Positive").sum())
    negative = int((result["Sentiment"] == "Negative").sum())
    neutral = int((result["Sentiment"] == "Neutral").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Feedback", total)
    c2.metric("Positive", positive)
    c3.metric("Negative", negative)
    c4.metric("Neutral", neutral)

    counts = result["Sentiment"].value_counts().reindex(
        ["Positive", "Negative", "Neutral"], fill_value=0
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of comments")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Sentiment Proportions")
        fig, ax = plt.subplots()
        counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Analyzed Feedback")
    st.dataframe(result, use_container_width=True)

    csv_data = result.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download Analyzed CSV",
        data=csv_data,
        file_name="analyzed_student_feedback.csv",
        mime="text/csv"
    )

    st.caption(
        "Method: rule-based sentiment analysis using predefined word lists. "
        "It is a simple educational baseline, not a trained ML model, "
        "and may misclassify context, sarcasm, or unfamiliar wording."
    )