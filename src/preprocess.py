import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps=PorterStemmer()
p=re.compile(r'[^a-zA-Z0-9\s]')

def preprocess(text):
    text=text.lower()
    text=re.sub(p,'',text)
    text=re.sub(r'\b(not|never|no)\s+(\w+)',r'\1_\2',text)
    tokens=text.split()
    stemmed=[ps.stem(word) for word in tokens]
    return ' '.join(stemmed)