import urllib
import urllib.request
import csv
import io
import argparse
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('Required Arguments')
requiredNamed.add_argument('--url', dest='url', metavar='url',
                           help='Webpage URL: eg-http://mappings.dbpedia.org/server/ontology/classes/Place', required=True)
args = parser.parse_args()

quote_page = args.url
page = urllib.request.urlopen(quote_page)

soup = BeautifulSoup(page, "html.parser")
# print type(soup)
fl = 0
cnt = 0
data = []
place_labels = []
for rows in soup.find_all("tr"):

    x = rows.find_all("td")

    if len(x) <= 2:
        fl = 1
        continue

    if fl == 1:
        fl = 2
        continue

    name = rows.find_all("td")[0].get_text().replace(" (edit)", "")
    label = rows.find_all("td")[1].get_text()
    dom = rows.find_all("td")[2].get_text()
    rng = rows.find_all("td")[3].get_text()

    dp = [name, label, dom, rng]
    place_labels.append(name)
    data.append(dp)

with io.open("get_properties.csv", mode='w', encoding='UTF8', newline='') as toWrite:
    writer = csv.writer(toWrite)
    writer.writerows(data)

with io.open("place_labels.csv", mode='w', encoding='UTF8', newline='') as toWrite:
    writer = csv.writer(toWrite)
    writer.writerows(place_labels)
