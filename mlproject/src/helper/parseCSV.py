from pprint import pprint
import csv
from dateutil.parser import parse
from dataStructureHelper import *

kvStore = {}
statStore = {}
ID = 0

def getOccurence(ID):
    global statStore
    return statStore[ID]

def getString(ID):
    global kvStore
    for key, value in kvStore.items():
        if value == ID:
            return value

def getID(string):
    global statStore, kvStore, ID
    # TODO: could be interesting to switch key and value so it make more sense
    # kvStore.get(string) == NONE
    # kvStore[id] = string
    if string not in kvStore:
        statStore[ID] = 0
        kvStore[string] = ID
        ID += 1
    else:
        statStore[kvStore[string]] += 1
    return kvStore[string]

def parseSeiyuu(filename):
    results = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, quotechar = '"', delimiter = ',', quoting = csv.QUOTE_ALL, skipinitialspace = True)
        for row in reader:
            results.append(row)
    return results

def parseStaff(filename):
    results = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, quotechar = '"', delimiter = ',', quoting = csv.QUOTE_ALL, skipinitialspace = True)
        for row in reader:
            row[JOB] = row[JOB].split(',')
            for i, job in enumerate(row[JOB]):
                row[JOB][i] = getID(job)
                results.append(row)
    return results

def parseLicensor(filename):
    results = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, quotechar = '"', delimiter = ',', quoting = csv.QUOTE_ALL, skipinitialspace = True)
        for row in reader:
            row[NAME] = getID(row[NAME])
            results.append(row)
            # getString(row[NAME]) getOccurence(row[NAME]) # to get string value and number of occurence
    return results

def parseAnime(filename):
    results = []
    i = 0
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, quotechar = '"', delimiter = ',', quoting = csv.QUOTE_ALL, skipinitialspace = True)
        for row in reader:
            title = []
            title.append(row[TITLE])
            row[TITLE] = "x"
            pprint(title)
            row[TYPE] = getID(row[TYPE])
            row[SOURCE] = getID(row[SOURCE])
            pprint(row[DATE])
            row[DATE] = parse(row[DATE])
            row[DATE] = row[DATE].year
            row[GENRE] = row[GENRE].strip(' ')[:-1].split(',')
            for i, genre in enumerate(row[GENRE]):
                row[GENRE][i] = getID(genre)
            row[DURATION] = getID(row[DURATION])
            row[RATING] = getID(row[RATING])
            results.append(row)
        writer = open("clean_anime.csv", "w+")
        for things in results:
            i = 0
            for i, stuff in enumerate(things):
                if (isinstance(stuff, list)):
                    writer.write("\"" + str(stuff).strip("[']") + "\"")
                else:
                    writer.write(str(stuff))
                if i < len(things) - 1:
                    writer.write(", ")
            writer.write("\n");
    return results

pprint("#######  ANIME   #######")
#pprint(parseAnime("anime.csv"))
parseAnime("../../../jikanAPI/jikan/anime.csv")
pprint("#######  LICENSOR   #######")
#pprint(parseLicensor("licensor.csv"))
pprint("#######  STUDIO   #######")
#pprint(parseLicensor("studio.csv"))
pprint("#######  LICENSOR   #######")
#pprint(parseLicensor("licensor.csv"))
pprint("#######  PRODUCER   #######")
#pprint(parseLicensor("producer.csv"))
pprint("#######  STAFF   #######")
#pprint(parseStaff("staff.csv"))
pprint("#######  VA   #######")
#pprint(parseSeiyuu("voice_actor.csv"))
