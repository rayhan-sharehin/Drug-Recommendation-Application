import pandas as pd
import numpy as np
import joblib
import gradio as gr

# Load model, vectorizer, and label encoder
model = joblib.load("random_forest_best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset
df = pd.read_csv("preprocessed_drug_reviews.csv")
df.dropna(subset=["review_cleaned"], inplace=True)
df["encoded_category"] = label_encoder.transform(df["category"])
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

def recommend_drug_with_rating(condition, review_text, top_n=3):
    X_input = vectorizer.transform([review_text])
    predicted_label = model.predict(X_input)[0]
    predicted_category = label_encoder.inverse_transform([predicted_label])[0]

    filtered_df = df[
        (df["condition"].str.lower() == condition.lower()) & 
        (df["encoded_category"] == predicted_label)
    ]

    if filtered_df.empty:
        return f"❌ No drugs found for '{condition}' with predicted sentiment: {predicted_category}"

    drug_summary = (
        filtered_df.groupby("drugName")
        .agg(
            count=("drugName", "count"),
            avg_rating=("rating", "mean"),
            sample_review=("review_cleaned", "first")
        )
        .sort_values(by=["avg_rating", "count"], ascending=False)
        .head(top_n)
    )

    output = f"🔍 Predicted Sentiment: {predicted_category}\n\n"
    output += f"✅ Top {top_n} Drug Recommendations for '{condition}':\n\n"
    for idx, row in drug_summary.iterrows():
        output += f"💊 {idx}\n⭐ Avg Rating: {row['avg_rating']:.2f} ({int(row['count'])} reviews)\n📝 Review: {row['sample_review'][:100]}...\n\n"
    return output

iface = gr.Interface(
    fn=recommend_drug_with_rating,
    inputs=[
        gr.Textbox(label="Condition", placeholder="e.g. Depression"),
        gr.Textbox(label="Your Review", placeholder="Describe how you feel..."),
        gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Top N Recommendations")
    ],
    outputs="text",
    title="💊 Drug Recommendation System with Rating & Side Effects",
    description="Enter your condition and experience to get the best drug recommendations based on patient reviews."
)

if __name__ == "__main__":
    iface.launch()
