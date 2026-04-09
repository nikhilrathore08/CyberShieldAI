import re
import nltk
from nltk.corpus import stopwords
## temp cahnge
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) ->str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+","",text)
    text = re.sub(r"[^a-z\s]", "", text)
 #   text = re.sub(r"\s+","",text).strip()

    tokens = [
        word for word in text.split()
        if word not in STOP_WORDS and len(word) > 2
    ]

    return " ".join(tokens)