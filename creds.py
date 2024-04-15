api_key="rterfdgdfgdgdf"
#---
import re
import pandas as pd

def count_sql_attributes(query):
    # Define the SQL clauses and join types
    clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN', 'WITH']

    # Initialize a dictionary to store the counts and attributes
    counts = {clause: 0 for clause in clauses}
    attributes = {clause: [] for clause in clauses}

    # Count the number of attributes in each clause
    for clause in clauses:
        if clause in query.upper():
            if clause == 'SELECT':
                select_parts = re.findall(r'(?i)(?<=SELECT)(.*?)(?=FROM|$)', query, re.DOTALL)
                for select_part in select_parts:
                    select_attributes = [i.strip().split(' AS ')[-1] for i in select_part.split(',') if i.strip()]
                    attributes['SELECT'].append(select_attributes)
                    counts['SELECT'] += len(select_attributes)
            elif clause == 'FROM':
                from_part = query.split('WHERE')[0].split('FROM')[1].strip()
                from_attributes = [i.strip() for i in from_part.split(',') if i.strip()]
                attributes['FROM'].extend(from_attributes)
                counts['FROM'] += len(from_attributes)
            elif clause == 'WHERE':
                where_part = query.split('WHERE')[1].split('GROUP BY')[0].strip()
                attributes['WHERE'].append(where_part)
                counts['WHERE'] += 1
            elif clause == 'GROUP BY':
                group_by_part = query.split('GROUP BY')[1].split('HAVING')[0].strip()
                group_by_attributes = [attr.strip() for attr in group_by_part.split(',')]
                attributes['GROUP BY'].extend(group_by_attributes)
                counts['GROUP BY'] += len(group_by_attributes)
            elif clause == 'HAVING':
                condition = re.search(r'(?i)\b' + clause + r'\b\s*(.+?)(?=\bGROUP BY\b|$)', query)
                if condition:
                    condition_text = condition.group(1).strip()
                    attributes[clause].append(condition_text)
                    counts[clause] += 1
            elif clause == 'WITH':
                cte_part = query.split('SELECT')[0].strip()
                cte_name = re.search(r'(?i)\bAS\b\s*([^\s]+)', cte_part).group(1)
                cte_select_parts = re.findall(r'(?i)(?<=SELECT)(.*?)(?=FROM|$)', cte_part, re.DOTALL)
                cte_attributes = []
                for cte_select_part in cte_select_parts:
                    cte_select_attributes = [i.strip().split(' AS ')[-1] for i in cte_select_part.split(',') if i.strip()]
                    cte_attributes.append(cte_select_attributes)
                    counts['SELECT'] += len(cte_select_attributes)
                attributes['WITH'].append({cte_name: cte_attributes})
                counts['WITH'] += 1

    # Extract join clauses and their attributes
    join_types = ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN']
    for join_type in join_types:
        join_parts = re.findall(r'(?i)\b' + join_type + r'\b\s*(.*?)\bON\b', query)
        for part in join_parts:
            tables_with_aliases = re.findall(r'\b[a-zA-Z]+\b', part)
            tables = [table.strip() for i, table in enumerate(tables_with_aliases) if i % 2 == 0 and table.strip()]
            attributes[join_type].extend(tables)
            counts[join_type] += len(tables)

    # Convert the counts to a DataFrame
    df = pd.DataFrame(list(counts.items()), columns=['Attribute', 'Count'])
    df['Attributes'] = list(attributes.values())
    return df

query = """WITH avg_per_store AS
  (SELECT store, AVG(amount) AS average_order
   FROM orders
   GROUP BY store)
SELECT o.id, o.store, o.amount, avg.average_order AS avg_for_store
FROM orders o
JOIN avg_per_store avg
ON o.store = avg.store;"""

query1 = """SELECT D.name AS DepartmentName, AVG(E.salary) AS AverageSalary, COUNT(E.id) AS NumberOfEmployees, L.location AS Location
    ...: FROM (SELECT * FROM Department WHERE dept_id=32) D
    ...: INNER JOIN Employee E ON D.id = E.department_id
    ...: OUTER JOIN Location L ON D.location_id = L.id
         LEFT JOIN ROUTE R ON R.Place= E.Place
    ...: WHERE E.hire_date > '2020-01-01' AND R.ROUTE="Houtrori"
    ...: GROUP BY D.name, L.location
    ...: HAVING COUNT(E.id) > 5 AND AVG(E.salary) > 50000;"""

df = count_sql_attributes(query1)
print(df)

#---
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
