import sys
import io
import csv

f = open('data/sparql_generator.csv', 'r')

lines = f.readlines()
f.close()
fl = 1

output = ""
for line in lines:

    if fl:
        fl = 0
        continue
    l = line.split(',')
    # print l

    newl, to_remove = [], []
    newl.append("dbo:Place")
    newl.append("")
    newl.append("")

    nlq = l[6].split()
    for i in range(len(nlq)):
        if '(' in nlq[i] or ')' in nlq[i]:
            to_remove.append(nlq[i])
            continue
        if '<' not in nlq[i] and '?' not in nlq[i]:
            nlq[i] = nlq[i].lower()

    for x in to_remove:
        nlq.remove(x)

    spq = l[-2].split()
    for i in range(len(spq)):
        if '<' not in spq[i] and '?' not in spq[i]:
            spq[i] = spq[i].lower()

    gq = l[-1].split()
    for i in range(len(gq)):
        if '<' not in gq[i] and '?' not in gq[i] and '[' not in gq[i]:
            gq[i] = gq[i].lower()

    newl.append(" ".join(nlq))
    newl.append(" ".join(spq))
    newl.append(" ".join(gq))
    output += ";".join(newl) + "\n"


fw = open('data/final_formatting_v1.csv', 'w')
fw.write(output)
fw.close()
