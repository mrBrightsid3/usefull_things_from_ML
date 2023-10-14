from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, "pkl_objects", "stopwords.pkl"), "rb"))


def tokenizer(text):
    text = re.sub("<[^>]*>", "", text)  # удаление html разметки
    emoticons = re.findall(
        "(?::|;|=)(?:-)?(?:-\)|\(|D|Р)", text
    )  # находми все эмотиконы
    text = re.sub(
        "[\W]+", " ", text.lower()
    ) + " ".join(  # находим и удалям все несловарные символы
        emoticons
    ).replace(
        "-", ""
    )  # возвращаем эмотиконы
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


vect = HashingVectorizer(
    decode_error="ignore", n_features=2**21, preprocessor=None, tokenizer=tokenizer
)
