from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from Qlearning import State_key, Q

import numpy as np

from collections import defaultdict
from math import log
import re

def TLS_dataset():
    from cesnet_datazoo.datasets import CESNET_TLS22
    from cesnet_datazoo.config import DatasetConfig, AppSelection, ValidationApproach
    
    dataset = CESNET_TLS22(data_root="~/datasets/CESNET-TLS22/", size="XS")
    
    common_params = {
        "dataset" : dataset,
        "test_period_name" : "W-2021-41",
        "val_approach": ValidationApproach.SPLIT_FROM_TRAIN,
        "train_val_split_fraction": 0.2
    }

    dataset_config = DatasetConfig(**common_params)
    dataset.set_dataset_config_and_initialize(dataset_config)
    train_dataframe = dataset.get_train_df(flatten_ppi=True)
    val_dataframe = dataset.get_val_df(flatten_ppi=True)
    test_dataframe = dataset.get_test_df(flatten_ppi=True)

    return (train_dataframe, val_dataframe, test_dataframe)

def QUIC_dataset():
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection, ValidationApproach

    dataset = CESNET_QUIC22("~/datasets/CESNET-QUIC22/", size="XS")

    common_params = {
        "dataset": dataset,
        "apps_selection": AppSelection.ALL_KNOWN,
        "train_period_name": "W-2022-44",
        "val_approach": ValidationApproach.SPLIT_FROM_TRAIN,
        "train_val_split_fraction": 0.2,
        "use_packet_histograms": True,
    }

    dataset_config = DatasetConfig(**common_params)
    dataset.set_dataset_config_and_initialize(dataset_config)
    train_dataframe = dataset.get_train_df(flatten_ppi=True)
    val_dataframe = dataset.get_val_df(flatten_ppi=True)
    test_dataframe = dataset.get_test_df(flatten_ppi=True)

    return (train_dataframe, val_dataframe, test_dataframe)

# same amount of samples from all classes
def create_balanced_test_data(nfeatures):
    grouped = test_dataframe.groupby('APP')

    X_arr = np.ndarray(shape = (10100, nfeatures))
    y_arr = np.ndarray(shape = (10100,))

    for index, i in enumerate(grouped):
        X_temp = i[1].drop(columns="APP").to_numpy()
        y_temp = i[1]["APP"].to_numpy()

        X_arr[index*100:(index*100)+100] = X_temp[:100]
        y_arr[index*100:(index*100)+100] = y_temp[:100]

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
            
    def client_bytes_into_discrete(self, nbytes):
        return int(np.clip(int(log(nbytes, 3)) - 5, 1, 9) - 1)
    
    def server_bytes_into_discrete(self, nbytes):
        return int(np.clip(int(log(nbytes, 5)) - 4, 0, 7))


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
        self.used = 1

        self.X_used = np.ndarray(shape=(iters, nfeatures))
        self.y_used = np.ndarray(shape=(iters,))
        self.cur_i = 0

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

        # print(key)

# bytes_client=1bytes_server=1duration=0percent_duration=2percent_of_class=2ppi_duration=0ppi_percent_duration=2
# bytes_client=1bytes_server=1duration=1percent_duration=3percent_of_class=3ppi_duration=1ppi_percent_duration=2
# bytes_client=3bytes_server=1duration=1percent_duration=3percent_of_class=1ppi_duration=0ppi_percent_duration=2
# bytes_client=1bytes_server=0duration=1percent_duration=3percent_of_class=0ppi_duration=0ppi_percent_duration=2
# bytes_client=1bytes_server=3duration=0percent_duration=2percent_of_class=1ppi_duration=0ppi_percent_duration=2
# bytes_client=2bytes_server=1duration=2percent_duration=2percent_of_class=2ppi_duration=3ppi_percent_duration=0
# bytes_client=2bytes_server=1duration=1percent_duration=3percent_of_class=2ppi_duration=1ppi_percent_duration=2
# bytes_client=2bytes_server=1duration=2percent_duration=2percent_of_class=2ppi_duration=3ppi_percent_duration=0
# bytes_client=1bytes_server=1duration=0percent_duration=2percent_of_class=2ppi_duration=0ppi_percent_duration=2
# bytes_client=2bytes_server=1duration=1percent_duration=3percent_of_class=1ppi_duration=0ppi_percent_duration=2
# bytes_client=1bytes_server=1duration=1percent_duration=3percent_of_class=1ppi_duration=0ppi_percent_duration=2
# bytes_client=2bytes_server=1duration=1percent_duration=3percent_of_class=1ppi_duration=0ppi_percent_duration=2
# bytes_client=2bytes_server=3duration=1percent_duration=3percent_of_class=2ppi_duration=0ppi_percent_duration=2

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

            self.duration_amount[prev_duration] += 1
            self.ppi_duration_amount[prev_ppi_duration] += 1
            self.class_amount[self.y_used[index - 1]] += 1
            self.used += 1

        self.cur_state["class_percent"] = self.class_amount[next_class] / self.used

        self.cur_state["bytes_client"] = self.client_bytes_into_discrete(X[index][90])
        self.cur_state["bytes_server"] = self.server_bytes_into_discrete(X[index][91])

        self.cur_state["duration"] = self.duration_into_discrete(self.X[index][94])
        duration_percent = self.duration_amount[self.cur_state["duration"]] / self.used
        self.cur_state["percent_duration"] = self.duration_into_discrete(duration_percent)

        self.cur_state["ppi_duration"] = self.ppi_duration_into_discrete(self.X[index][97])
        ppi_duration_percent = self.ppi_duration_amount[self.cur_state["ppi_duration"]] / self.used
        self.cur_state["ppi_percent_duration"] = self.ppi_percent_duration_into_discrete(ppi_duration_percent)

    def state_to_key(self):
        key = ""
        for state_var in self.used_state:
            key += state_var + "=" + str(self.cur_state[state_var])

        return key

    def isTake(self):
        key = self.state_to_key()

        if key in self.state_action:
            print("is in")
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
        
        clf = RandomForestClassifier()
        clf.fit(self.X[:self.cur_i], self.y[:self.cur_i])
        
        predict_arr = clf.predict(X_test)
        
        print(f"q_learning_acc: {accuracy_score(y_test, predict_arr):.4f}" + "\n")

        val = 0
        for i in range(3):
            clf = RandomForestClassifier()
            indices = np.random.choice(iters, self.cur_i, replace=False)
    
            clf.fit(self.X[indices], self.y[indices])
            
            predict_arr = clf.predict(X_test)

            val += accuracy_score(y_test, predict_arr)

        val /= 3
        print(f"random_learning_acc: {val:.4f}" + "\n")
        

        clf = RandomForestClassifier()
        clf.fit(self.X[:iters], self.y[:iters])
        
        predict_arr = clf.predict(X_test)
        
        print(f"total_learning_acc: {accuracy_score(y_test, predict_arr):.4f}" + "\n")


if __name__ == '__main__':
    (train_dataframe, val_dataframe, test_dataframe) = QUIC_dataset()

    X = train_dataframe.drop(columns="APP").to_numpy()
    y = train_dataframe["APP"].to_numpy()

    X_test = test_dataframe.drop(columns="APP").to_numpy()
    y_test = test_dataframe["APP"].to_numpy()

    iters = 10000
    nfeatures = X.shape[1]

    q_tester = Qlearning_tester(iters, nfeatures)

    q_tester.load_q_df("out_state.txt")

    q_tester.X = X
    q_tester.y = y
    q_tester.test(iters)

    q_tester.test_acc(iters, X_test, y_test)

# if __name__ == '__main__':
#     ##### qlearning params ######
#     increased_rd = 500 # increased randomness for n iters
#     decrease_alpha = 0.0001
#     iters = 100
#     base_samples_amount = 0
#     epsilon = 0.9
#     alpha = 0.2
#     gamma = 0.9
#     #############################

#     q = Q()
    
#     (train_dataframe, val_dataframe, test_dataframe) = QUIC_dataset()

#     q.X = train_dataframe.drop(columns="APP").to_numpy()
#     q.y = train_dataframe["APP"].to_numpy()

#     q.X_big_test = test_dataframe.drop(columns="APP").to_numpy()[:100000]
#     q.y_big_test = test_dataframe["APP"].to_numpy()[:100000]

#     nfeatures = q.X.shape[1]

#     (q.X_test, q.y_test) = create_balanced_test_data(nfeatures)

#     q.initialize(nfeatures, iters, base_samples_amount, epsilon, alpha, gamma)

#     state_key = q.update_state(State_key(0, 0, 0, 0, 0, 0, 0), 0)
#     q.learn(state_key, iters, increased_rd)

#     q.save_table_to_file()