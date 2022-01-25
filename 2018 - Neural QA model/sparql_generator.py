import io
import csv

f = open('data/mve_and_oe_v2.csv', 'r')
lines = f.readlines()

data = []
lineno = 1
for line in lines:
    if lineno == 1:
        lineno += 1
        continue
    line = line.strip().split(',')
    if line[4] != '':
        st = 'SELECT ?x WHERE { <A> <' + line[4] + '> ?x }'
        gqt = 'SELECT ?a WHERE { ?a <' + line[4] + '> [] . ?a a <http://dbpedia.org/ontology/Place> }'

    dp = [line[0], line[1], line[2], line[3],
          line[4], line[5], line[6], line[7], st, gqt]
    data.append(dp)


with io.open("data/sparql_generator.csv", mode='w', encoding='UTF8', newline='') as toWrite:
    writer = csv.writer(toWrite)
    writer.writerow(["Name", "Label", "Domain", "Range", "URI", "Count", "MVE", "OE", "Sparql-Template", "Generator-Query-Template"])
    writer.writerows(data)
