from pprint import pprint
import csv

results = []
with open("test.csv") as csvfile:
    reader = csv.reader(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
    for row in reader:
        results.append(row)
pprint(results)
