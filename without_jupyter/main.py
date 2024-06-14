from sklearn.ensemble import RandomForestClassifier

from Qlearning import State_key, Q

import numpy as np

import re

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

    def __init__(self, iters, nfeatures):
        self.pattern = re.compile(r"State_key\((.*?)\) (\d) ([\d\.\-e]+)")
        self.pattern_q = re.compile("qvalues")
        self.pattern_r = re.compile("rewards")

        self.state_action = {}
        self.cur_state = {}

        self.X_used = np.ndarray(shape=(iters, nfeatures))
        self.y_used = np.ndarray(shape=(iters,))
        self.cur_i = 0

        self.used_state = None


    def parse_single_q(self, state, action, value):
        state_key_parts = state.split(', ')
        state_key_parts.sort()

        if self.used_state == None:
            self.used_state = []

            for el in state_key_parts:
                el = el.split("=")[0]
                self.used_state.append(el)


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

    def state_to_key(self):
        key = ""
        for state_var in self.used_state:
            key += state_var + "=" + str(self.cur_state[state_var])

        return key


    def isTake(self):
        key = self.state_to_key(self.cur_state)

        if key in self.state_action:
            return self.state_action[key][0]
        else:
            # State that was not seen in training
            # Probably very valuable sample
            return 1

    def test(self, X, y):
        for index, el in enumerate(X):
            if self.isTake():
                self.X_used[self.cur_i] = el
                self.y_used[self.cur_i] = y[index]

                self.update_state(X)


if __name__ == '__main__':
    (train_dataframe, val_dataframe, test_dataframe) = QUIC_dataset()

    X = train_dataframe.drop(columns="APP").to_numpy()
    y = train_dataframe["APP"].to_numpy()

    iters = 10000
    nfeatures = X.shape[1]

    q_tester = Qlearning_tester(iters, nfeatures)

    q_tester.load_q_df("out_state.txt")

    q_tester.test()


# if __name__ == '__main__':
#     ##### qlearning params ######
#     increased_rd = 500 # increased randomness for n iters
#     decrease_alpha = 0.0001
#     iters = 100
#     base_samples_amount = 400
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

#     state_key = q.update_state(State_key(0, 0, 0, 0, 0), 0)
#     q.learn(state_key, iters, increased_rd)

#     q.save_table_file()