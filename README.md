# Drug-Recommendation-Application
This project is an interactive real-time drug recommendation application that leverages machine learning and natural language processing to classify the sentiment of drug reviews and recommend the most effective medications for a given medical condition.

Key Features:

Sentiment Classification: Utilizes a fine-tuned transformer model (DistilBERT) from Hugging Face to analyze the tone of patient reviews and classify them as Positive or Adverse.

Personalized Recommendations: Suggests drugs based on both the predicted sentiment and the medical condition described by the user, ensuring high relevance and user-specific matching.

Rich User Insights: Displays average drug ratings and representative review excerpts to enhance transparency for decision-making.

Side Effect Search: Allows users to filter recommendations by side effects mentioned in real reviews for more informed choices.

Robust Data Handling: Processes and cleans a large real-world dataset (61,000+ reviews), filtering out noisy and irrelevant data for improved accuracy.

User-Friendly Interface: Features a Gradio-powered web app with an intuitive dropdown search and real-time responses.

Technologies Used:

Python, Pandas: Data wrangling and preprocessing.

scikit-learn, Transformers (Hugging Face): Sentiment analysis and ML modeling.

Gradio: Building an interactive front-end for user engagement.

Kaggle Drug Review Dataset: Comprehensive review dataset for analysis.

Applications for Pharmaceuticals:

This system supports pharmacovigilance, drug efficacy studies, and patient-centric recommendations by harnessing real-world, user-generated data. The interactive platform can be further adapted for clinical decision support, customer feedback analysis, and patient engagement workflows.

Live Demo and Source Code:
Explore the application and code repository here: [[[GitHub Project Link](https://github.com/rayhan-sharehin/Drug-Recommendation-Application.git)](https://huggingface.co/spaces/sharehin/drug-recommendation-app)]
