import pandas as pd
import numpy as np
from nltk.tokenize import casual_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.utils import resample
from sklearn.utils import shuffle 
from collections import Counter
from collections import OrderedDict
import string
from keras import backend as K
### NLP Preprocessing ###

'''
Here we want to construct a complete nlp preprocessing pipeline so that we can take a corpus of documents and squeeze the real value out of it. 

1. Stopwords - It is important to get rid of frequently occuring words like 'the' and 'a' that don't tell us very much about the nature of the document
2. Stemming - Many words have very similar core meaning, but are different perhaps in their number or tense. Stemming will collapse such words into a single token
3. Term Frequency / Inverse Document Frequency(TF-IDF) - In order to really understand the significance that a word holds in a document, we have to see how many times it comes up in a document, and then measure that against how many times that word comes up across all the docs in the corpus. Higher scores generally mean more meaningful words. 
'''

sw = stopwords.words('english')
sw.remove('not')
sw.remove('no')


def remove_stopwords(doc: list or str, join: bool = False) -> list or str:
    if not join:
        if isinstance(doc, list):
            filtered_tokens = [w for w in doc if w not in sw]
            return filtered_tokens
        elif isinstance(doc, str):
            doc = doc.split()
            filtered_tokens = [w for w in doc if w not in sw]
            return filtered_tokens
        else:
            raise TypeError("argument doc takes type lst or type str")
    elif join:
        if isinstance(doc, list):
            filtered_tokens = [w for w in doc if w not in sw]
            return " ".join(filtered_tokens)
        elif isinstance(doc, str):
            doc = doc.split()
            filtered_tokens = [w for w in doc if w not in sw]
            return filtered_tokens
        else:
            raise TypeError("argument doc takes type lst or type str")
        
    

def tf(document: str or list) -> dict:
    if isinstance(document, str):
        document = document.split()
        
    elif isinstance(document, list):
        pass
    
    else:
        raise TypeError
        
    bow = dict(Counter(document))
    for word, count in bow.items():
        bow[word] = count/len(document)
    return bow

def idf(corpus: list) -> dict:
    output = []
    vocab = set()
    for doc in corpus:
        for word in doc:
            vocab.add(word)

    vocabulary = {word: 0 for word in vocab}
    
    f_counter = 0
    for word, value in vocabulary.items():
        doc_count = 0

        for i, doc in enumerate(corpus):
            if word in doc:
                doc_count += 1
                

        vocabulary[word] = np.log((len(corpus)/float(doc_count)))
        f_counter += 1
        if f_counter % 1000 == 1:
            print(f'Number of words IDFed: {f_counter}')
        
    return vocabulary

def tf_idf(stemmed_corpus: list) -> dict:
    output = []
    
    tfed = [tf(doc) for doc in stemmed_corpus]
    idfed = idf(stemmed_corpus)
    
    doc_counter = 0
    for doc in tfed:
        for word in doc:
            doc[word] *= idfed[word]
        
        doc_counter += 1
#         if doc_counter % 1000 == 0:
#             print(f"Vectorised {doc_counter} documents")
        
    return tfed

def resample_majority(df: pd.DataFrame, criteria: str, majority_class: int) -> pd.DataFrame:
    majority_df = df[df[criteria] == majority_class]
    minority_df = df[df[criteria] != majority_class]
    minority_mean = int(minority_df[criteria].value_counts().mean())
    resampled_majority = resample(majority_df, replace = False, n_samples = minority_mean, random_state = 42)
    reassembled_df = shuffle(pd.concat([resampled_majority, minority_df]), random_state=42)
    return reassembled_df.reset_index(drop=True)


### Metrics for finer evaluation of keras models ###

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

### Verbose evaluation of keras models ###

def verbose_evaluate(model, X_test, y_test):
    metrics = model.metrics_names
    metric_scores = model.evaluate(X_test, y_test)
    score_dict = {}
    for i, metric in enumerate(metric_scores):
        print(f'The {metrics[i]} score is: {metric}')
        score_dict.update({metrics[i]: metric})
    return score_dict