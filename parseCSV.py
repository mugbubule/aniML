from pprint import pprint
import csv
from dateutil.parser import parse
from dataStructureHelper import *

kvStore = {}
ID = 0

def getID(string):
    global kvStore, ID
    if string not in kvStore:
        kvStore[string] = ID
        ID += 1
    return kvStore[string]

def parseAnime(filename):
        results = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, quotechar = '"', delimiter = ',', quoting = csv.QUOTE_ALL, skipinitialspace = True)
            for row in reader:
                row[TYPE] = getID(row[TYPE])
                row[SOURCE] = getID(row[SOURCE])
                row[DATE] = parse(row[DATE])
                row[GENRE] = row[GENRE].strip(' ')[:-1].split(',')
                for i, genre in enumerate(row[GENRE]):
                    row[GENRE][i] = getID(genre)
                row[DURATION] = getID(row[DURATION])
                row[RATING] = getID(row[RATING])
                results.append(row)
        pprint(results)

parseAnime("anime.csv")
