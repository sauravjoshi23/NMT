import urllib
import urllib.request
import http.client
import json
import sys
import pandas as pd

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
    print("QUERY: ", query)
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

    resp = urllib.request.urlopen(endpoint + "?" + urllib.parse.urlencode(param))
    j = resp.read()
    resp.close()
    j = json.loads(j)['results']['bindings'][0]['count']['value']
    sys.stdout.flush()
    return j


filename = 'place_labels.csv'
df = pd.read_csv(filename)
for index, row in df.iterrows():
    print(row)
    # q1 = create_query("abstract", "ontology")
    # q2 = create_query("abstract", "property")
    # result1 = sparql_query(q1)
    # result2 = sparql_query(q2)
    # mx = max(result1, result2)
    # print(result1, result2)
