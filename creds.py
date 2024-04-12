api_key="rterfdgdfgdgdf"
import re
import pandas as pd

def count_sql_attributes(query):
    # Define the SQL clauses
    clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'JOIN', 'INNER JOIN', 'OUTER JOIN', 'HAVING']

    # Initialize a dictionary to store the counts
    counts = {clause: 0 for clause in clauses}
    attributes = {clause: [] for clause in clauses}

    # Count the number of attributes in each clause
    for clause in clauses:
        if clause in query:
            if clause == 'SELECT':
                select_part = query.split('FROM')[0].split('SELECT')[1]
                attributes['SELECT'] = [i.strip() for i in select_part.split(',') if i.strip()]
                counts['SELECT'] = len(attributes['SELECT'])
            elif clause in ['JOIN', 'INNER JOIN', 'OUTER JOIN']:
                join_part = query.split('ON')[0].split(clause)[1]
                attributes[clause] = [i.strip() for i in join_part.split(',') if i.strip()]
                counts[clause] = len(attributes[clause])
            else:
                counts[clause] = 1

    # Convert the counts to a DataFrame
    df = pd.DataFrame(list(counts.items()), columns=['Attribute', 'Count'])
    df['Attributes'] = list(attributes.values())
    return df

query = "SELECT ty.yui, ty.iuyt, ty.oiu,ru.kj FROM trise tr INNER JOIN ruise ru ON ru.id=tr.id"
df = count_sql_attributes(query)
print(df)





import sqlparse
import pandas as pd

def count_sql_attributes(query):
    parsed = sqlparse.parse(query)[0]
    tokens = parsed.tokens
    attributes = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'JOIN']
    counts = {attr: 0 for attr in attributes}

    for token in tokens:
        if token.ttype is None and str(token) in attributes:
            if str(token) == 'SELECT':
                select_list = str(token.get_parent()).split('SELECT')[1].split('FROM')[0]
                counts['SELECT'] = len([i.strip() for i in select_list.split(',') if i.strip()])
            else:
                counts[str(token)] += 1

    df = pd.DataFrame(list(counts.items()), columns=['Attribute', 'Count'])
    return df

query = "SELECT ty.yui, ty.iuyt, ty.oiu,ru.kj FROM trise tr JOIN ruise ru ON ru.id=tr.id"
df = count_sql_attributes(query)
print(df)







import re
import pandas as pd

def count_sql_attributes(query):
    # Define the SQL clauses
    clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'JOIN', 'INNER JOIN', 'OUTER JOIN', 'HAVING']

    # Initialize a dictionary to store the counts
    counts = {clause: 0 for clause in clauses}

    # Count the number of attributes in each clause
    for clause in clauses:
        if clause in query:
            if clause == 'SELECT':
                select_part = query.split('FROM')[0].split('SELECT')[1]
                counts['SELECT'] = len([i.strip() for i in select_part.split(',') if i.strip()])
            elif clause in ['JOIN', 'INNER JOIN', 'OUTER JOIN']:
                join_part = query.split('ON')[0].split(clause)[1]
                counts[clause] = len([i.strip() for i in join_part.split(',') if i.strip()])
            else:
                counts[clause] = 1

    # Convert the counts to a DataFrame
    df = pd.DataFrame(list(counts.items()), columns=['Attribute', 'Count'])
    return df

query = "SELECT ty.yui, ty.iuyt, ty.oiu,ru.kj FROM trise tr INNER JOIN ruise ru ON ru.id=tr.id"
df = count_sql_attributes(query)
print(df)



import re

def extract_sql(text):
    pattern = r'```(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

text = """This is the final extracted data for the query asked by the user uer. This is the hive query ```SELECT * FROM Br_Premium_data``` This query may not the be most suitable one please check your attributes correctly"""

print(extract_sql(text))


import pandas as pd

def add_column_using_function(df, func, new_col_name):
    """
    Add a new column to the DataFrame using a provided function.

    Parameters:
    df (DataFrame): The DataFrame to add a new column to.
    func (function): The function to apply to each row of the DataFrame.
    new_col_name (str): The name of the new column.

    Returns:
    DataFrame: The DataFrame with the new column added.
    """
    df[new_col_name] = df.apply(func, axis=1)
    return df


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
