api_key="rterfdgdfgdgdf"

from nltk.util import ngrams
def rouge_n(reference, hypothesis, n):

    ref_ngrams = ngrams(reference, n)

    hyp_ngrams = ngrams(hypothesis, n)
 
    common = ref_ngrams.intersection(hyp_ngrams)

    return len(common) / len(ref_ngrams)

from nltk.translate.bleu_score import sentence_bleu
def bleu_score(reference, hypothesis):
    return sentence_bleu([reference], hypothesis)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
def cos_sim(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
