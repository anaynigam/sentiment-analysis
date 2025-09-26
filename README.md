CUSTOMER REVIEW SENTIMENT ANALYSIS

A web application that analyzes customer reviews and predicts whether they are 'NEGATIVE','POSITIVE' or 'NEUTRAL'.
Built with python, streamlit, and machine learning this app helps business and individuals quickly gauge customer sentiment from text feedback.

TRY IT LIVE 🚀

[SENTIMENT ANALYSIS OF CUSTOMER REVIEW]-(https://anaynigam-sentiment-analysis-app-aumxwl.streamlit.app/)

🛠️ Features
- predicts "sentiment of customer reviews" in real-time.
  
- Preprocessing Handles:
   -Test cleaning and normalization
   -negation handling like "not good" or "not bad"
   -Tokenization
  
-Uses a "Naive Bayes Classifier" with TF-IDF vectorization.
-Incorporates "synthetic data" to improve accuracy.
-Displays "metrics" including accuracy, f1 score, confusion matrix.

💻 Tech Stack

-python
-pandas
-scikit-learn
-streamlit
-Pickle

TO RUN LOCALLY
1.
---

## ⚙️ How to Run Locally
1. Clone the repo:
git clone https://github.com/anaynigam/sentiment-analysis.git
cd sentiment-analysis

2. Create a Virtual Environment
   python -m venv .venv

3. Activate the Virtual Environment
   Windows: .\.venv\Scripts\activate
   Linux/macOS: source .venv/bin/activate

4. Run Streamlit App
   streamlit run app.py
   
📝 License

This project is licensed under the MIT License.
