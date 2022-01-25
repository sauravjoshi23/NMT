import urllib
import urllib.request
import csv
import io
import json
import sys
import pandas as pd
from tqdm import tqdm

endpoint = "http://dbpedia.org/sparql"
graph = "http://dbpedia.org"

Q = dict()
Q["query"] = "SELECT (count(distinct ?s) as ?count) WHERE { ?s mydbo [] . } "

def create_query(curr_class, type):

    query = Q["query"]
    if type == 'ontology':
        query = query.replace("mydbo", "dbo:" + curr_class)
    elif type == 'property':
        query = query.replace("mydbo", "dbp:" + curr_class)
    #print("QUERY: ", query)
    return query

def sparql_query(query):
    param = dict()
    # param["default-graph-uri"] = graph
    param["query"] = query
    # print param["query"]
    param["format"] = "JSON"
    param["CXML_redir_for_subjs"] = "121"
    param["CXML_redir_for_hrefs"] = ""
    param["timeout"] = "36000"  # ten minutes - works with Virtuoso endpoints
    param["debug"] = "on"
    try:
        resp = urllib.request.urlopen(endpoint + "?" + urllib.parse.urlencode(param))
        j = resp.read()
        resp.close()
        j = json.loads(j)['results']['bindings'][0]['count']['value']
        sys.stdout.flush()
        return j
    except:
        return 0


filename = 'data/get_properties.csv'
df = pd.read_csv(filename)
data = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    name = row[0]
    q1 = create_query(name, "ontology")
    # q2 = create_query("abstract", "property")
    result1 = sparql_query(q1)
    # result2 = sparql_query(q2)
    # mx = max(result1, result2)
    uri = "http://dbpedia.org/ontology/" + name 
    dp = [uri, result1]
    data.append(dp)

with io.open("data/properties_count.csv", mode='w', encoding='UTF8', newline='') as toWrite:
    writer = csv.writer(toWrite)
    writer.writerows(data)
