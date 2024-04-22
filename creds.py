api_key="rterfdgdfgdgdf"

from rouge_score import rouge_scorer

def calculate_rouge_scores(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    scores = scorer.score(' '.join(reference), ' '.join(hypothesis))
    
    rouge1 = scores['rouge1'].fmeasure
    rouge2 = scores['rouge2'].fmeasure
    rougeL = scores['rougeL'].fmeasure
    rougeLsum = scores['rougeLsum'].fmeasure
    
    return rouge1, rouge2, rougeL, rougeLsum

# Example usage:
reference = ["The cat sat on the mat"]
hypothesis = ["The cat sat on the rug"]
rouge1, rouge2, rougeL, rougeLsum = calculate_rouge_scores(reference, hypothesis)
print("ROUGE-1:", rouge1)
print("ROUGE-2:", rouge2)
print("ROUGE-L:", rougeL)
print("ROUGE-Lsum:", rougeLsum)

## 16/04/25 19:50
import re
import pandas as pd

def count_sql_attributes(query):
    # Define the SQL clauses and join types
    clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN']

    # Initialize a dictionary to store the counts and attributes
    counts = {clause: 0 for clause in clauses}
    attributes = {clause: [] for clause in clauses}

    # Convert query to uppercase for case-insensitive matching
    query_upper = query.upper()

    # Count the number of attributes in each clause
    for clause in clauses:
        if clause in query_upper:
            if clause == 'SELECT':
                select_parts = re.findall(r'(?i)(?<=SELECT)(.*?)(?=FROM|$)', query, re.DOTALL)
                for select_part in select_parts:
                    select_attributes = [i.strip().split(' AS ')[-1] for i in select_part.split(',') if i.strip()]
                    attributes['SELECT'].append(select_attributes)
                    counts['SELECT'] += len(select_attributes)
            elif clause == 'FROM':
                from_part = re.search(r'(?i)FROM\s+(.*?)(?=\bWHERE|\bGROUP BY|\bHAVING|\bORDER BY|\bUNION|\b$)', query, re.DOTALL).group(1).strip()
                from_attributes = from_part.split()[0]
                attributes['FROM'].append(from_attributes)
                counts['FROM'] += 1
            elif clause == 'WHERE':
                where_part = re.search(r'(?i)\bWHERE\b\s+(.*?)(?=\bGROUP BY\b|\bHAVING\b|\bORDER BY\b|\bUNION\b|$)', query, re.DOTALL)
                if where_part:
                    where_conditions = where_part.group(1).strip().split('AND')
                    where_conditions = [condition.strip() for condition in where_conditions]
                    attributes['WHERE'].extend(where_conditions)
                    counts['WHERE'] += len(where_conditions)
            elif clause == 'GROUP BY':
                group_by_part = re.search(r'(?i)\bGROUP BY\b\s+(.*?)(?=\bHAVING\b|$)', query, re.DOTALL)
                if group_by_part:
                    group_by_attributes = [attr.strip() for attr in group_by_part.group(1).split(',')]
                    attributes['GROUP BY'].extend(group_by_attributes)
                    counts['GROUP BY'] += len(group_by_attributes)
            elif clause == 'HAVING':
                having_part = re.search(r'(?i)\bHAVING\b\s+(.*?)(?=\bORDER BY\b|\bUNION\b|$)', query, re.DOTALL)
                if having_part:
                    having_conditions = having_part.group(1).strip().split('AND')
                    having_conditions = [condition.strip() for condition in having_conditions]
                    attributes['HAVING'].extend(having_conditions)
                    counts['HAVING'] += len(having_conditions)
            elif clause in ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN']:
                join_parts = re.findall(r'(?i)\b' + clause + r'\b\s+(.*?)\bON\b', query, re.DOTALL)
                for part in join_parts:
                    tables_with_aliases = re.findall(r'(\w+\.\w+)', part)
                    tables = [table.strip() for table in tables_with_aliases]
                    attributes['JOIN'].extend(tables)
                    counts['JOIN'] += len(tables)

    # Convert the counts to a DataFrame
    df = pd.DataFrame(list(counts.items()), columns=['Attribute', 'Count'])
    df['Attributes'] = list(attributes.values())
    # Remove duplicates from the 'JOIN' attribute
    df.loc[df['Attribute'] == 'JOIN', 'Attributes'] = df.loc[df['Attribute'] == 'JOIN', 'Attributes'].apply(lambda x: list(set(x)))
    df.loc[df['Attribute'] == 'JOIN', 'Count'] = len(df.loc[df['Attribute'] == 'JOIN', 'Attributes'].values[0])
    return df


query_lc = """select d.name as departmentname, avg(e.salary) as averagesalary, count(e.id) as numberofemployees, l.location as location
from ascend.department d
join Ascend.employee e on d.id = e.department_id
outer join Ascend.location l on d.location_id = l.id
left join Ascend.route r on r.place= e.place
where e.hire_date > '2020-01-01' and r.route="houtrori"
group by d.name, l.location
having count(e.id) > 5 and avg(e.salary) > 50000;"""

df_lc = count_sql_attributes(query_lc)
print(df_lc)

print("--------------")

query_uc = """SELECT D.name AS DepartmentName, AVG(E.salary) AS AverageSalary, COUNT(E.id) AS NumberOfEmployees, L.location AS Location
FROM ascend.Department D
INNER JOIN Ascend.Employee E ON D.id = E.department_id
OUTER JOIN Ascend.Location L ON D.location_id = L.id
LEFT JOIN Ascend.ROUTE R ON R.Place= E.Place
WHERE E.hire_date > '2020-01-01' AND R.ROUTE="Houtrori"
GROUP BY D.name, L.location
HAVING COUNT(E.id) > 5 AND AVG(E.salary) > 50000;"""

df_uc = count_sql_attributes(query_uc)
print(df_uc)


#---
# 16/04/25 17:53
import re
import pandas as pd

def count_sql_attributes(query):
    # Define the SQL clauses and join types
    clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING','JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN']

    # Initialize a dictionary to store the counts and attributes
    counts = {clause: 0 for clause in clauses}
    attributes = {clause: [] for clause in clauses}

    # Convert query to uppercase for case-insensitive matching
    query_upper = query.upper()

    # Count the number of attributes in each clause
    for clause in clauses:
        if clause in query_upper:
            if clause == 'SELECT':
                select_parts = re.findall(r'(?i)(?<=SELECT)(.*?)(?=FROM|$)', query, re.DOTALL)
                for select_part in select_parts:
                    select_attributes = [i.strip().split(' AS ')[-1] for i in select_part.split(',') if i.strip()]
                    attributes['SELECT'].append(select_attributes)
                    counts['SELECT'] += len(select_attributes)
            elif clause == 'FROM':
                from_part = re.search(r'(?i)FROM\s+(.*?)(?=\bINNER JOIN|\bLEFT JOIN|\bRIGHT JOIN|\bOUTER JOIN|\bWHERE|\bGROUP BY|\bHAVING|$)', query, re.DOTALL).group(1).strip()
                from_subquery_match = re.search(r'(?i)\((.*?)\)', from_part)
                if from_subquery_match:
                    from_attributes = from_subquery_match.group(1).strip()
                else:
                    from_attributes = from_part.split('FROM')[-1].strip()

                # Extract table alias, if present
                from_alias_match = re.search(r'\b(\w+)\s+AS\b', from_attributes)
                if from_alias_match:
                    from_alias = from_alias_match.group(1)
                    attributes['FROM'].append(from_alias)
                else:
                    attributes['FROM'].append(from_attributes)
                counts['FROM'] += 1

            elif clause == 'WHERE':
                where_part = re.search(r'(?i)\bWHERE\b\s+(.*?)(?=\bGROUP BY\b|\bHAVING\b|\bORDER BY\b|\bUNION\b|$)', query, re.DOTALL)
                if where_part:
                    where_conditions = where_part.group(1).strip().split('AND')
                    where_conditions = [condition.strip() for condition in where_conditions]
                    attributes['WHERE'].extend(where_conditions)
                    counts['WHERE'] += len(where_conditions)
            elif clause == 'GROUP BY':
                group_by_part = re.search(r'(?i)\bGROUP BY\b\s+(.*?)(?=\bHAVING\b|$)', query, re.DOTALL)
                if group_by_part:
                    group_by_attributes = [attr.strip() for attr in group_by_part.group(1).split(',')]
                    attributes['GROUP BY'].extend(group_by_attributes)
                    counts['GROUP BY'] += len(group_by_attributes)
            elif clause == 'HAVING':
                having_part = re.search(r'(?i)\bHAVING\b\s+(.*?)(?=\bORDER BY\b|\bUNION\b|$)', query, re.DOTALL)
                if having_part:
                    having_conditions = having_part.group(1).strip().split('AND')
                    having_conditions = [condition.strip() for condition in having_conditions]
                    attributes['HAVING'].extend(having_conditions)
                    counts['HAVING'] += len(having_conditions)

    # Extract join clauses and their attributes
    join_types = ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN']
    for join_type in join_types:
        join_parts = re.findall(r'(?i)\b' + join_type + r'\b\s*(.*?)\bON\b', query, re.DOTALL)
        for part in join_parts:
            tables_with_aliases = re.findall(r'\b[a-zA-Z]+\b', part)
            tables = [table.strip() for i, table in enumerate(tables_with_aliases) if i % 2 == 0 and table.strip()]
            attributes[join_type].extend(tables)
            counts[join_type] += len(tables)

    # Convert the counts to a DataFrame
    df = pd.DataFrame(list(counts.items()), columns=['Attribute', 'Count'])
    df['Attributes'] = list(attributes.values())
    return df




query_lc = """select d.name as departmentname, avg(e.salary) as averagesalary, count(e.id) as numberofemployees, l.location as location
from department d
join employee e on d.id = e.department_id
outer join location l on d.location_id = l.id
left join route r on r.place= e.place
where e.hire_date > '2020-01-01' and r.route="houtrori"
group by d.name, l.location
having count(e.id) > 5 and avg(e.salary) > 50000;"""

df_lc = count_sql_attributes(query_lc)
print(df_lc)

print("--------------")

query_uc = """SELECT D.name AS DepartmentName, AVG(E.salary) AS AverageSalary, COUNT(E.id) AS NumberOfEmployees, L.location AS Location
FROM Department D
INNER JOIN Employee E ON D.id = E.department_id
OUTER JOIN Location L ON D.location_id = L.id
LEFT JOIN ROUTE R ON R.Place= E.Place
WHERE E.hire_date > '2020-01-01' AND R.ROUTE="Houtrori"
GROUP BY D.name, L.location
HAVING COUNT(E.id) > 5 AND AVG(E.salary) > 50000;"""

df_uc = count_sql_attributes(query_uc)
print(df_uc)

#---
#16/04/24
import re
import pandas as pd

def count_sql_attributes(query):
    # Define the SQL clauses and join types
    clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN']

    # Initialize a dictionary to store the counts and attributes
    counts = {clause: 0 for clause in clauses}
    attributes = {clause: [] for clause in clauses}

    # Convert query to uppercase for case-insensitive matching
    query_upper = query.upper()

    # Count the number of attributes in each clause
    for clause in clauses:
        if clause in query_upper:
            if clause == 'SELECT':
                select_parts = re.findall(r'(?i)(?<=SELECT)(.*?)(?=FROM|$)', query, re.DOTALL)
                for select_part in select_parts:
                    select_attributes = [i.strip().split(' AS ')[-1] for i in select_part.split(',') if i.strip()]
                    attributes['SELECT'].append(select_attributes)
                    counts['SELECT'] += len(select_attributes)
            elif clause == 'FROM':
                from_part = re.search(r'(?i)FROM\s+(.*?)(?=\bINNER JOIN|\bLEFT JOIN|\bRIGHT JOIN|\bOUTER JOIN|\bWHERE|\bGROUP BY|\bHAVING|$)', query, re.DOTALL).group(1).strip()
                from_subquery_match = re.search(r'(?i)\((.*?)\)', from_part)
                if from_subquery_match:
                    from_attributes = from_subquery_match.group(1).strip()
                else:
                    from_attributes = from_part.split('FROM')[-1].strip()

                # Extract table alias, if present
                from_alias_match = re.search(r'\b(\w+)\s+AS\b', from_attributes)
                if from_alias_match:
                    from_alias = from_alias_match.group(1)
                    attributes['FROM'].append(from_alias)
                else:
                    attributes['FROM'].append(from_attributes)
                counts['FROM'] += 1

            elif clause == 'WHERE':
                where_part = re.search(r'(?i)\bWHERE\b\s+(.*?)(?=\bGROUP BY\b|\bHAVING\b|\bORDER BY\b|\bUNION\b|$)', query, re.DOTALL)
                if where_part:
                    where_conditions = where_part.group(1).strip().split('AND')
                    where_conditions = [condition.strip() for condition in where_conditions]
                    attributes['WHERE'].extend(where_conditions)
                    counts['WHERE'] += len(where_conditions)
            elif clause == 'GROUP BY':
                group_by_part = re.search(r'(?i)\bGROUP BY\b\s+(.*?)(?=\bHAVING\b|$)', query, re.DOTALL)
                if group_by_part:
                    group_by_attributes = [attr.strip() for attr in group_by_part.group(1).split(',')]
                    attributes['GROUP BY'].extend(group_by_attributes)
                    counts['GROUP BY'] += len(group_by_attributes)
            elif clause == 'HAVING':
                having_part = re.search(r'(?i)\bHAVING\b\s+(.*?)(?=\bORDER BY\b|\bUNION\b|$)', query, re.DOTALL)
                if having_part:
                    having_conditions = having_part.group(1).strip().split('AND')
                    having_conditions = [condition.strip() for condition in having_conditions]
                    attributes['HAVING'].extend(having_conditions)
                    counts['HAVING'] += len(having_conditions)

    # Extract join clauses and their attributes
    join_types = ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN']
    for join_type in join_types:
        join_parts = re.findall(r'(?i)\b' + join_type + r'\b\s*(.*?)\bON\b', query, re.DOTALL)
        for part in join_parts:
            tables_with_aliases = re.findall(r'\b[a-zA-Z]+\b', part)
            tables = [table.strip() for i, table in enumerate(tables_with_aliases) if i % 2 == 0 and table.strip()]
            attributes[join_type].extend(tables)
            counts[join_type] += len(tables)

    # Convert the counts to a DataFrame
    df = pd.DataFrame(list(counts.items()), columns=['Attribute', 'Count'])
    df['Attributes'] = list(attributes.values())
    return df

#----

import re
import pandas as pd

def count_sql_attributes(query):
    # Define the SQL clauses and join types
    clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN']

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
                from_part = re.search(r'(?i)FROM\s+(.*?)(?=\bINNER JOIN|\bLEFT JOIN|\bRIGHT JOIN|\bOUTER JOIN|\bWHERE|\bGROUP BY|\bHAVING|$)', query, re.DOTALL).group(1).strip()
                from_subquery_match = re.search(r'(?i)\((.*?)\)', from_part)
                if from_subquery_match:
                    from_attributes = from_subquery_match.group(1).strip()
                else:
                    from_attributes = from_part.split('FROM')[-1].strip()

                # Extract table alias, if present
                from_alias_match = re.search(r'\b(\w+)\s+AS\b', from_attributes)
                if from_alias_match:
                    from_alias = from_alias_match.group(1)
                    attributes['FROM'].append(from_alias)
                else:
                    attributes['FROM'].append(from_attributes)
                counts['FROM'] += 1

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

query = """SELECT D.name AS DepartmentName, AVG(E.salary) AS AverageSalary, COUNT(E.id) AS NumberOfEmployees, L.location AS Location
FROM (SELECT * FROM Department WHERE dept_id=12) D
INNER JOIN Employee E ON D.id = E.department_id
OUTER JOIN Location L ON D.location_id = L.id
LEFT JOIN ROUTE R ON R.Place= E.Place
WHERE E.hire_date > '2020-01-01' AND R.ROUTE="Houtrori"
GROUP BY D.name, L.location
HAVING COUNT(E.id) > 5 AND AVG(E.salary) > 50000;"""

df = count_sql_attributes(query)
print(df)

#--
import re
import pandas as pd

def count_sql_attributes(query):
    # Define the SQL clauses and join types
    clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN']

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
                from_part = re.search(r'(?i)FROM\s+(.*?)(?=\bINNER JOIN|\bLEFT JOIN|\bRIGHT JOIN|\bOUTER JOIN|\bWHERE|\bGROUP BY|\bHAVING|$)', query, re.DOTALL).group(1).strip()
                from_subquery_match = re.search(r'(?i)\((.*?)\)', from_part)
                if from_subquery_match:
                    from_attributes = from_subquery_match.group(1).strip()
                else:
                    from_attributes = from_part.split('FROM')[-1].strip()

                # Extract table alias, if present
                from_alias_match = re.search(r'\b(\w+)\s+AS\b', from_attributes)
                if from_alias_match:
                    from_alias = from_alias_match.group(1)
                    attributes['FROM'].append(from_alias)
                else:
                    attributes['FROM'].append(from_attributes)
                counts['FROM'] += 1

            elif clause == 'WHERE':
                where_part = re.search(r'(?i)\bWHERE\b\s+(.*?)(?=\bGROUP BY\b|\bHAVING\b|\bORDER BY\b|\bUNION\b|$)', query, re.DOTALL).group(1).strip()
                where_conditions = re.findall(r'\b\w+\b\s*[!=><]+\s*[\'"]?[\w\s]*[\'"]?', where_part)
                attributes['WHERE'].extend([condition.strip() for condition in where_conditions])
                counts['WHERE'] += len(where_conditions)
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
query = """SELECT D.name AS DepartmentName, AVG(E.salary) AS AverageSalary, COUNT(E.id) AS NumberOfEmployees, L.location AS Location
FROM (SELECT * FROM Department WHERE dept_id=32) D
INNER JOIN Employee E ON D.id = E.department_id
OUTER JOIN Location L ON D.location_id = L.id
LEFT JOIN ROUTE R ON R.Place= E.Place
WHERE E.hire_date > '2020-01-01' AND R.ROUTE="Houtrori"
GROUP BY D.name, L.location
HAVING COUNT(E.id) > 5 AND AVG(E.salary) > 50000;"""

df = count_sql_attributes(query)
print(df)

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
