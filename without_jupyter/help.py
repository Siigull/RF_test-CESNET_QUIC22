from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np

from collections import defaultdict
from math import log
import re

def topx_indexes(dataframe, nclasses):
    grouped_counts = dataframe.groupby("APP").size()
    grouped_counts = grouped_counts.sort_values(ascending=False)

    topx_groups = grouped_counts.head(nclasses).index

    return topx_groups

def TLS_dataset(nclasses = 0):
    from cesnet_datazoo.datasets import CESNET_TLS22
    from cesnet_datazoo.config import DatasetConfig, AppSelection, ValidationApproach
    
    dataset = CESNET_TLS22(data_root="~/datasets/CESNET-TLS22/", size="XS")

    common_params = {
        "dataset" : dataset,
        "apps_selection" : AppSelection.ALL_KNOWN,
        "test_period_name" : "W-2022-44",
        "val_approach": ValidationApproach.SPLIT_FROM_TRAIN,
        "train_val_split_fraction": 0.2
    }

    dataset_config = DatasetConfig(**common_params)
    dataset.set_dataset_config_and_initialize(dataset_config)
    train_dataframe = dataset.get_train_df(flatten_ppi=True)
    val_dataframe = dataset.get_val_df(flatten_ppi=True)
    test_dataframe = dataset.get_test_df(flatten_ppi=True)

    if nclasses != 0:
        topx_groups = topx_indexes(train_dataframe, nclasses)

        train_dataframe = train_dataframe[train_dataframe["APP"].isin(topx_groups)]
        test_dataframe  = test_dataframe[test_dataframe["APP"].isin(topx_groups)]
        val_dataframe   = val_dataframe[val_dataframe["APP"].isin(topx_groups)]

    return (train_dataframe, val_dataframe, test_dataframe)

def QUIC_dataset(nclasses = 0):
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection, ValidationApproach

    dataset = CESNET_QUIC22("~/datasets/CESNET-QUIC22/", size="XS")

    common_params = {
        "dataset" : dataset,
        "apps_selection" : AppSelection.ALL_KNOWN,
        "test_period_name" : "W-2022-44",
        "val_approach": ValidationApproach.SPLIT_FROM_TRAIN,
        "train_val_split_fraction": 0.2
    }

    dataset_config = DatasetConfig(**common_params)
    dataset.set_dataset_config_and_initialize(dataset_config)
    train_dataframe = dataset.get_train_df(flatten_ppi=True)
    val_dataframe = dataset.get_val_df(flatten_ppi=True)
    test_dataframe = dataset.get_test_df(flatten_ppi=True)

    if nclasses != 0:
        topx_groups = topx_indexes(train_dataframe, nclasses)

        train_dataframe = train_dataframe[train_dataframe["APP"].isin(topx_groups)]
        test_dataframe  = test_dataframe[test_dataframe["APP"].isin(topx_groups)]
        val_dataframe   = val_dataframe[val_dataframe["APP"].isin(topx_groups)]

    return (train_dataframe, val_dataframe, test_dataframe)

# same amount of samples from all classes
def create_balanced_test_data(nfeatures, test_dataframe, nfrom_class = 100):
    grouped = test_dataframe.groupby("APP")

    X_arr = np.ndarray(shape = (nfrom_class * len(grouped), nfeatures))
    y_arr = np.ndarray(shape = (nfrom_class * len(grouped),))

    for index, i in enumerate(grouped):
        X_temp = i[1].drop(columns="APP").to_numpy()
        y_temp = i[1]["APP"].to_numpy()

        X_arr[index*nfrom_class:(index * nfrom_class) + nfrom_class] = X_temp[:nfrom_class]
        y_arr[index*nfrom_class:(index * nfrom_class) + nfrom_class] = y_temp[:nfrom_class]

    return (X_arr, y_arr)

class Qlearning_tester():
    def value_into_discrete(self, value, thresholds):
        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return i
            
        return len(thresholds)

    def class_percent_into_discrete(self, percent):
        return self.value_into_discrete(percent, self.CLASS_PERCENT_VALUES)

    def predict_proba_into_discrete(self, proba):
        return self.value_into_discrete(proba, self.PREDICT_PROBA_VALUES)

    def percent_used_into_discrete(self, percent):
        return self.value_into_discrete(percent, self.PERCENT_USED_VALUES)
    
    def duration_into_discrete(self, duration):
        return self.value_into_discrete(duration, self.DURATION_VALUES)

    def percent_duration_into_discrete(self, duration_percent):
        return self.value_into_discrete(duration_percent, self.DURATION_PERCENT_VALUES)
    
    def ppi_duration_into_discrete(self, duration):
        return self.value_into_discrete(duration, self.PPI_DURATION_VALUES)
    
    def ppi_percent_duration_into_discrete(self, duration_percent):
        return self.value_into_discrete(duration_percent, self.PPI_DURATION_PERCENT_VALUES)
    
    def centroid_size_into_discrete(self, size):
        return self.value_into_discrete(size, self.CENTROID_SIZE_VALUES)
            
    def client_bytes_into_discrete(self, nbytes):
        return int(np.clip(int(log(nbytes, 3)) - 5, 1, 9) - 1)
    
    def server_bytes_into_discrete(self, nbytes):
        return int(np.clip(int(log(nbytes, 5)) - 4, 0, 7))

    def calculate_distance(self, sum, value, n):
        if n == 0:
            return value

        avg = sum / n

        return abs(avg - value)

    def calculate_centroid_size(self, cl, values):
        sum = 0
        for i in range(self.num_of_sizes):
            sum += self.calculate_distance(self.class_size_sums[cl][i], values[i + 60], self.to_i)

        return sum / self.num_of_sizes

    def __init__(self, iters, nfeatures):
        self.pattern = re.compile(r"State_key\((.*?)\) (\d) ([\d\.\-e]+)")
        self.pattern_q = re.compile("qvalues")
        self.pattern_r = re.compile("rewards")

        self.state_action = {}
        self.cur_state = {}

        self.duration_amount = defaultdict(int)
        self.ppi_duration_amount = defaultdict(int)
        self.class_amount = defaultdict(int)

        self.CLASS_PERCENT_VALUES        = [0.0001, 0.01, 0.05, 0.1]
        # self.PREDICT_PROBA_VALUES        = [0.25, 0.50, 0.75]
        self.DURATION_VALUES             = [0.1, 1, 29.9, 59.9, 89.9, 119.9, 299]
        self.DURATION_PERCENT_VALUES     = [0.05, 0.1, 0.2, 0.4]
        self.PPI_DURATION_VALUES         = [0.2, 9.9, 19.9, 70, 112]
        self.PPI_DURATION_PERCENT_VALUES = [0.005, 0.01, 0.1, 0.4]
        self.CENTROID_SIZE_VALUES        = [300, 500, 800]
        self.used = 1

        self.X_used = np.ndarray(shape=(iters, nfeatures))
        self.y_used = np.ndarray(shape=(iters,))
        self.cur_i = 0

        self.num_of_sizes = 12

        self.class_size_sums = {}
        for class_i in range(200):
            self.class_size_sums[class_i] = []
            for _ in range(self.num_of_sizes):
                self.class_size_sums[class_i].append(0)

        self.last_action = 0

        self.used_state = None


    def parse_single_q(self, state, action, value):
        state_key_parts = state.split(', ')
        state_key_parts.sort()

        if self.used_state == None:
            self.used_state = []

            for el in state_key_parts:
                el = el.split("=")[0]
                self.used_state.append(el)
                self.cur_state[el] = 0

        key = "".join(state_key_parts)

        if key not in self.state_action or \
           self.state_action[key][1] < value:
                self.state_action[key] = [action, value]


    def load_q_df(self, file_path):
        state = 0

        with open(file_path, "r") as f:
            for line in f:
                match = self.pattern_q.match(line)
                if match:
                    state = 1

                match = self.pattern_r.match(line)
                if match:
                    state = 2
                    break

                if state == 1:
                    match = self.pattern.match(line)

                    if match:
                        self.parse_single_q(match.group(1), 
                                            match.group(2), 
                                            match.group(3))

    def update_state(self, index):
        next_class = self.y[index]

        if self.last_action == 1:
            prev_duration = self.cur_state["duration"]
            prev_ppi_duration = self.cur_state["ppi_duration"]
            prev_class = self.y[index - 1]

            for i in range(self.num_of_sizes):
                self.class_size_sums[prev_class][i] = self.X_used[self.cur_i - 1][i + 60]

            self.duration_amount[prev_duration] += 1
            self.ppi_duration_amount[prev_ppi_duration] += 1
            self.class_amount[prev_class] += 1
            self.used += 1

        self.cur_state["percent_of_class"] \
            = self.class_percent_into_discrete(self.class_amount[next_class] / self.used)
        
        self.cur_state["bytes_client"] = self.client_bytes_into_discrete(self.X[index][90])
        self.cur_state["bytes_server"] = self.server_bytes_into_discrete(self.X[index][91])

        self.cur_state["duration"] = self.duration_into_discrete(self.X[index][94])
        duration_percent = self.duration_amount[self.cur_state["duration"]] / self.used
        self.cur_state["percent_duration"] = self.percent_duration_into_discrete(duration_percent)

        self.cur_state["ppi_duration"] = self.ppi_duration_into_discrete(self.X[index][97])
        ppi_duration_percent = self.ppi_duration_amount[self.cur_state["ppi_duration"]] / self.used
        self.cur_state["ppi_percent_duration"] = self.ppi_percent_duration_into_discrete(ppi_duration_percent)

        self.cur_state["centroid_size"] = self.calculate_centroid_size(next_class, self.X[index])

    def state_to_key(self):
        key = ""
        for state_var in self.used_state:
            key += state_var + "=" + str(self.cur_state[state_var])

        return key

    def isTake(self):
        key = self.state_to_key()

        if key in self.state_action:
            return self.state_action[key][0]
        else:
            # State that was not seen in training
            # Probably very valuable sample
            return 1

    def test(self, iters):
        for index, el in enumerate(self.X[:iters]):
            self.update_state(index)

            if self.isTake() == 1:
                self.X_used[self.cur_i] = el
                self.y_used[self.cur_i] = self.y[index]

                self.cur_i += 1

                self.last_action = 1

            else:
                self.last_action = 0

    def test_acc(self, iters, X_test, y_test):
        print(str(self.cur_i) + "/" + str(iters))
        
        clf = RandomForestClassifier(max_depth=10, n_jobs=-1)
        clf.fit(self.X[:self.cur_i], self.y[:self.cur_i])
        
        predict_arr = clf.predict(X_test)
        
        print(f"q_learning_acc: {accuracy_score(y_test, predict_arr):.4f}" + "\n")

        val = 0
        for i in range(3):
            clf = RandomForestClassifier(max_depth=10, n_jobs=-1)
            indices = np.random.choice(iters, self.cur_i, replace=False)
    
            clf.fit(self.X[indices], self.y[indices])
            
            predict_arr = clf.predict(X_test)

            val += accuracy_score(y_test, predict_arr)

        val /= 3
        print(f"random_learning_acc: {val:.4f}" + "\n")
        

        clf = RandomForestClassifier(max_depth=10, n_jobs=-1)
        clf.fit(self.X[:iters], self.y[:iters])
        
        predict_arr = clf.predict(X_test)
        
        print(f"total_learning_acc: {accuracy_score(y_test, predict_arr):.4f}" + "\n")