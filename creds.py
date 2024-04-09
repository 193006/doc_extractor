api_key="rterfdgdfgdgdf"

from nltk.util import ngrams

def rouge_n(reference, hypothesis, n):
    # Convert the generator to a set
    ref_ngrams = set(ngrams(reference, n))
    hyp_ngrams = set(ngrams(hypothesis, n))

    # Calculate the intersection of the two sets
    common = ref_ngrams.intersection(hyp_ngrams)

    # Return the ratio of common ngrams to reference ngrams
    return len(common) / len(ref_ngrams)

from nltk.util import ngrams

def rouge_n_fmeasure(reference, hypothesis, n):
    # Convert the generator to a set
    ref_ngrams = set(ngrams(reference, n))
    hyp_ngrams = set(ngrams(hypothesis, n))

    # Calculate the intersection of the two sets
    common = ref_ngrams.intersection(hyp_ngrams)

    # Calculate precision, recall, and F-measure
    precision = len(common) / len(hyp_ngrams) if hyp_ngrams else 0
    recall = len(common) / len(ref_ngrams) if ref_ngrams else 0
    f_measure = 2 * precision * recall / (precision + recall) if precision + recall else 0

    # Return the F-measure
    return f_measure


def count_words_and_chars(sentence):
    # Count the number of words in the sentence
    word_count = len(sentence.split())

    # Count the number of characters in the sentence
    char_count = len(sentence)

    return word_count, char_count




from nltk.translate.bleu_score import sentence_bleu
def bleu_score(reference, hypothesis):
    return sentence_bleu([reference], hypothesis)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
def cos_sim(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
