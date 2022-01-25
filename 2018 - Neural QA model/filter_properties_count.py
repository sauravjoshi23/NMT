import csv
import io
import pandas as pd
from tqdm import tqdm

filename = 'data/merge1.csv'
df = pd.read_csv(filename)
data = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    count = row[5]
    if count != 0:
        data.append([row[0], row[1], row[2], row[3], row[4], row[5]])


with io.open("data/filter_properties_count.csv", mode='w', encoding='UTF8', newline='') as toWrite:
    writer = csv.writer(toWrite)
    writer.writerow(["Name", "Label", "Domain", "Range", "URI", "Count"])
    writer.writerows(data)



