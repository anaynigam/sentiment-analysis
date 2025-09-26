import random

def predict_sentiment(text,nb,tfidf,le,preprocess_fn):
    clean=preprocess_fn(text)
    vect=tfidf.transform([clean])
    pred=nb.predict(vect)[0]
    pred_label=le.inverse_transform([pred])[0]

    emoji={'positive':'😊','negative':'😡','neutral':'😐'}
    if pred_label=='negative':
        stars=random.randint(1,2)
    elif pred_label=='neutral':
        stars=3
    else:
        stars=random.randint(4,5)

    return pred_label,emoji.get(pred_label,''),stars
