test1 = 'data/Monument_300/data.en'
test2 = 'data/Monument_300_paraphrase/data.en'
test3 = 'data/Monument_300/data.sparql'
file1 = open(test1, 'r', encoding="utf8")
Lines1 = file1.readlines()
file2 = open(test2, 'r', encoding="utf8")
Lines2 = file2.readlines()
file3 = open(test3, 'r', encoding="utf8")
Lines3 = file3.readlines()

data1 = []
data2 = []
for i in range(len(Lines1)):
    if Lines1[i] in Lines2:
        data1.append(Lines1[i])
        data2.append(Lines3[i])

outputfile = open(('data/Monument_300/test_benchmark.en'), 'w', encoding="utf8")
for x in data1:
    outputfile.writelines(x)
outputfile.close()

outputfile2 = open(('data/Monument_300/test_benchmark.sparql'), 'w', encoding="utf8")
for x in data2:
    outputfile2.writelines(x)
outputfile2.close()


