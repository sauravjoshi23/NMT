import sys
import re
import io
import csv

f = open('data/filter_properties_count.csv', 'r')
lines = f.readlines()
final_lines = []

lineno = 1

data = []
for line in lines:
    if lineno == 1:
        lineno += 1
        continue
    #print(line)
    line = line.strip().split(',')
    rng = line[3].lower()
    lbl = line[1]
    if 'person' in rng:
        rng = "who"
    else:
        rng = "what"
    mve = rng + " is the " + lbl + " of <A>"
    oe = rng + " is the " + lbl + " of <A>"
    dp = [line[0], line[1], line[2], line[3], line[4], line[5], mve, oe]
    data.append(dp)

with io.open("data/mve_and_oe_v1.csv", mode='w', encoding='UTF8', newline='') as toWrite:
    writer = csv.writer(toWrite)
    writer.writerow(["Name", "Label", "Domain", "Range", "URI", "Count", "MVE", "OE"])
    writer.writerows(data)
