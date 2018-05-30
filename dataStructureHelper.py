from enum import Enum

ID = 0
TITLE = 1
TYPE = 2
SOURCE = 3
EPISODES = 4
DATE = 5
DURATION = 6
RATING = 7
SCORE = 8
RANK = 9
SCORED_BY = 10
POPULARITY = 11
MEMBERS = 12
FAVORITES = 13
RELATED = 14
GENRE = 15
WATCHING = 16
COMPLETED = 17
ON_HOLD = 18
DROPPED = 19
PLAN_TO_WATCH = 20
TOTAL = 21

class Anime(Enum):
    ID = 0
    TITLE = 1
    TYPE = 2
    SOURCE = 3
    EPISODES = 4
    AIRED = 5
    DURATION = 6
    RATING = 7
    SCORE = 8
    RANK = 9
    SCORED_BY = 10
    POPULARITY = 11
    MEMBERS = 12
    FAVORITES = 13
    RELATED = 14
    GENRE = 15
    WATCHING = 16
    COMPLETED = 17
    ON_HOLD = 18
    DROPPED = 19
    PLAN_TO_WATCH = 20
    TOTAL = 21

LI_NAME = 0
ANIME_ID = 1

class Licensor(Enum):
    NAME = 0
    ANIME_ID = 1

ID = 0
NAME = 1
SURNAME = 2
ANIME_ID = 3

class VA(Enum):
    ID = 0
    NAME = 1
    SURNAME = 2
    ANIME_ID = 3

JOB = 3
ANIME_ID = 4

class Staff(Enum):
    ID = 0
    NAME = 1
    SURNAME = 2
    JOB = 3
    ANIME_ID = 4
