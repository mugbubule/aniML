import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot


class Data:
    # Load the csv file into pandas dataset

    def add_studio(self):
        names = ["studio_name", "anime_id"]
        studio = pd.read_csv("../../jikanAPI/jikan/studio.csv", quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL,
                            skipinitialspace=True, parse_dates=True, header=None, names=names,
                            dtype={'studio_name': object, "anime_id": np.int32})
        studio_val = studio.groupby("studio_name").size().to_frame('studio_val');
        working_table = pd.merge(studio, studio_val, left_on='studio_name', right_on='studio_name')
        working_table = self.dataset.merge(working_table, left_index=True, right_on='anime_id')
        working_table = working_table.groupby('anime_id').sum()
        self.dataset = working_table
        print(working_table)
        #self.add_licensor()

    def add_staff(self):
        names = ["staff_id", "staff_name", "staff_surname", "staff_position", "anime_id"]
        staff = pd.read_csv("../../jikanAPI/jikan/staff.csv", quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL,
                            skipinitialspace=True, parse_dates=True, header=None, names=names,
                            dtype={"staff_id": np.int32, 'staff_name': object, 'staff_surname': object, 'staff_position': object})
        staff_val = staff.groupby("staff_name").size().to_frame('staff_val');
        working_table = pd.merge(staff, staff_val, left_on='staff_name', right_on='staff_name')
        working_table = self.dataset.merge(working_table, left_index=True, right_on='anime_id')
        working_table = working_table.groupby('anime_id').sum()
        self.dataset = working_table
        print(working_table)
        self.add_studio()

    def add_voice_actor(self):
        names = ["voice_actor_id", "voice_actor_name", "voice_actor_surname", "anime_id"]
        voice_actor = pd.read_csv("../../jikanAPI/jikan/voice_actor.csv", quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL,
                            skipinitialspace=True, parse_dates=True, header=None, names=names,
                            dtype={'voice_actor_name': object, 'voice_actor_surname': object})
        voice_actor_val = voice_actor.groupby("voice_actor_name").size().to_frame('voice_actor_val');
        print(voice_actor)
        working_table = pd.merge(voice_actor, voice_actor_val, left_on='voice_actor_name', right_on='voice_actor_name')
        print(working_table)
        working_table = self.dataset.merge(working_table, left_index=True, right_on='anime_id')
        print(working_table)
        working_table = working_table.groupby('anime_id').sum()
        self.dataset = working_table
        print(working_table)
        self.add_staff()

    def add_producer(self): # ça à l'air bon mais omg faut vérifier
        print(self.dataset)
        names = ["producer_name", "anime_id"]
        producer = pd.read_csv("../../jikanAPI/jikan/producer.csv", quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL,
                            skipinitialspace=True, parse_dates=True, header=None, names=names,
                            dtype={'producer_name': object, "anime_id": np.int32})
        producer_val = producer.groupby("producer_name").size().to_frame('producer_val');
        working_table = pd.merge(producer, producer_val, left_on='producer_name', right_on='producer_name')
        working_table = self.dataset.merge(working_table, left_index=True, right_on='anime_id')
        working_table = working_table.groupby('anime_id').sum()
        print(working_table)
        self.dataset = working_table
        self.add_voice_actor()

    def __init__(self, filename):
        self.names = ["id", "title", "type", "source", "episodes", "aired", "duration", "rating", "score", "rank",
                      "scored_by", "popularity", "members", "favorites", "related", "genre", "watching", "completed",
                      "on_hold", "dropped", "plan_to_watch", "total"]

        self.features_selected = ['type', 'source', 'score', 'episodes', 'aired', 'duration', 'rating', 'related']

        self.dataset = pd.read_csv(filename, index_col=['id'], quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL,
                                   skipinitialspace=True, header=None,  names=self.names,
                                   dtype={'title': str, 'type': np.int32, 'source': np.int32,
                                          'episodes': np.int32, 'aired': np.int32, 'duration': np.int32,
                                          'rating': np.int32, 'score': np.float64, 'rank': np.int32,
                                          'scored_by': np.int32, 'popularity': np.int32, 'members': np.int32,
                                          'favorites': np.int32, 'related': np.int32, 'genre': str,
                                          'watching': np.int32, 'completed': np.int32, 'on_hold': np.int32,
                                          'dropped': np.int32, 'plan_to_watch': np.int32, 'total': np.int32})

        print("\n============ DATASET TYPES ============")
        print(self.dataset.dtypes)
        print()
        self.compute_columns()
        self.add_producer()

    def compute_columns(self):
        self.dataset['watching_percent'] = self.dataset['watching'] / self.dataset['total']
        self.dataset['dropped_percent'] = self.dataset['dropped'] / self.dataset['total']
        self.dataset['completed_percent'] = self.dataset['completed'] / self.dataset['total']
        self.names.append("watching_percent");
        self.names.append("dropped_percent");
        self.names.append("completed_percent");

        self.features_selected.append("watching_percent");
        self.features_selected.append("dropped_percent");
        self.features_selected.append("completed_percent");

        print("\n============ NEW DATASET TYPES ============")
        print(self.dataset.dtypes)
        print()

    # Print correlations stat data
    def visualize(self):
        print("\n============ DATASET SIZE ============")
        print(self.dataset.shape[0], end="", flush=True), print(" rows")
        print(self.dataset.shape[1], end="", flush=True), print(" colunms")
        pd.set_option('precision', 1)

        print("\n============ DATASET GENERAL STATISTICS ============")
        print(self.dataset.describe())

        print("\n============ DATASET CORRELATIONS ============")
        pd.set_option('precision', 3)
        self.correlations = self.dataset[self.features_selected].corr(method='pearson')
        print(self.correlations)
        print()

    # Plot the data
    def plot(self):
        self.dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
        pyplot.show()
        self.dataset.plot(kind='density', subplots=True, layout=(5, 5), sharex=False, legend=True, fontsize=1)
        pyplot.show()
        self.dataset.plot(kind='box', subplots=True, layout=(5, 5), sharex=False, sharey=False, fontsize=8)
        pyplot.show()

    def plot_correlations(self):
        # plot correlation matrix
        fig = pyplot.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(self.features_selected), 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.features_selected)
        ax.set_yticklabels(self.features_selected)
        pyplot.show()
