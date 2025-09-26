import os
import pickle
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import preprocess
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score


model_file="model/nb_model.pkl"
tfidf_file="model/tfidf.pkl"
le_file="model/label_encoder.pkl"

def create_synthetic_data():
    positive_reviews = [
        "I love this product", "This is amazing", "Very satisfied with this purchase",
        "Highly recommend to everyone", "Works perfectly", "Excellent quality",
        "Exceeded my expectations", "Best purchase I have made", "Fantastic performance",
        "Really happy with it", "Battery life is excellent", "Screen quality is superb",
        "Comfortable and stylish", "Very easy to use", "Fast delivery and well packaged",
        "High value for money", "Sound quality is amazing", "Absolutely perfect for daily use",
        "Durable and reliable", "Perfect gift for friends"
    ]

    negative_reviews = [
        "I hate this product", "This is terrible", "Very disappointed with this purchase",
        "Do not recommend", "Stopped working quickly", "Poor quality", "Did not meet expectations",
        "Worst purchase ever", "Awful performance", "Really unhappy with it", "Battery drains fast",
        "Screen scratches easily", "Very uncomfortable", "Difficult to use", "Late delivery and damaged",
        "Overpriced for quality", "Sound quality is bad", "Broke after a week", "Fragile and weak",
        "Not worth the money"
    ]

    negation_reviews = [
        "Not good at all", "Not bad at all", "Never liked this product",
        "Does not work properly", "Not what I expected", "Not very satisfied",
        "Not recommended", "Not worth the money", "Not comfortable", "Not impressive"
    ]

    mixed_reviews = [
        "Good quality but delivery was late", "Not bad but could be better",
        "Battery life is great but camera is bad", "Comfortable but looks cheap",
        "Amazing performance yet a bit noisy", "Not perfect but I like it",
        "Pretty good but not excellent", "Fast delivery but poor packaging",
        "Loved it but found minor defects", "Great features but expensive"
    ]

    reviews, labels = [], []

    while len(reviews) < 500:
        r = random.choice(positive_reviews)
        reviews.append(r)
        labels.append("positive")

    while len(reviews) < 1000:
        r = random.choice(negative_reviews)
        reviews.append(r)
        labels.append("negative")

    for r in negation_reviews:
        reviews.append(r)
        if "not bad" in r.lower():
            labels.append("positive")
        else:
            labels.append("negative")

    for r in mixed_reviews:
        reviews.append(r)
        if any(word in r.lower() for word in ["good", "great", "amazing", "loved", "excellent"]):
            labels.append("positive")
        else:
            labels.append("negative")

    return pd.DataFrame({"detail": reviews, "Sentiment": labels})


def train_load_model(model_dir="models"):
    os.makedirs(model_dir,exist_ok=True)
    model_file=os.path.join(model_dir,"nb_model.pkl")
    vectorizer_file=os.path.join(model_dir,"tfidf.pkl")
    encoder_file=os.path.join(model_dir,"label_encoder.pkl")
    metrics_file=os.path.join(model_dir,"metrics.pkl")

    if all(os.path.exists(f) for f in [model_file,vectorizer_file,encoder_file]):
        nb=pickle.load(open(model_file,"rb"))
        tfidf=pickle.load(open(vectorizer_file,"rb"))
        le=pickle.load(open(encoder_file,"rb"))
        metrics=pickle.load(open(metrics_file,"rb"))


        return nb, tfidf, le, metrics
    BASE_DIR=os.path.dirname(__file__)
    data=os.path.join(BASE_DIR,"..","data","Equal.csv")
    df=pd.read_csv(data,encoding="latin1")
    df['detail']=df['Summary']+' '+df['Review']
    df=df[['detail','Sentiment']]

    df_syn=create_synthetic_data()
    df=pd.concat([df,df_syn],ignore_index=True)
    df=df.sample(frac=1,random_state=42).reset_index(drop=True)
    df['detail']=df['detail'].apply(preprocess)

    le=LabelEncoder()
    y=le.fit_transform(df['Sentiment'])

    tfidf=TfidfVectorizer(max_features=15000,ngram_range=(1,2))
    x=tfidf.fit_transform(df['detail'])

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    nb=MultinomialNB()
    nb.fit(x_train,y_train)

    y_pred=nb.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=le.classes_)
    }

    metrics_file=os.path.join(model_dir,"metrics.pkl")
    with open(metrics_file,"wb") as f:
            pickle.dump(metrics,f)


    pickle.dump(nb, open(model_file, "wb"))
    pickle.dump(tfidf, open(vectorizer_file, "wb"))
    pickle.dump(le, open(encoder_file, "wb"))

    return nb, tfidf, le, metrics