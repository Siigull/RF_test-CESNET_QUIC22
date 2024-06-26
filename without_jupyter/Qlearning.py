from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from pyqlearning.q_learning import QLearning

from collections import defaultdict
from dataclasses import dataclass, asdict
from math import log
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

used_features  = [1, 1, 1, 1, 1, 1, 1, 1]
classes_amount = [5, 8, 5, 9, 7, 6, 5, 4]

@dataclass
class State_key:
    percent_of_class     : int # 1e-2, 1e-1. 5e-1, 1, more than 1
    duration             : int # 8 buckets
    percent_duration     : int # 5 buckets
    bytes_client         : int # log3, 9 buckets
    bytes_server         : int # log5, 7 buckets
    ppi_duration         : int # 6
    ppi_percent_duration : int # 5
    centroid_size        : int # 4

    def write(self):
        ret = "State_key("

        for index, field in enumerate(self.__dataclass_fields__):
            if used_features[index]:
                ret += field + "=" + str(getattr(self, field)) + ", "

        ret = ret[:-2] # remove trailing ", "

        ret += ")"

        return ret

    def __hash__(self):
        value = 0
        multiplier = 1
        for index, el in enumerate(asdict(self).values()):
            if used_features[index]:
                value += el * multiplier
                multiplier *= classes_amount[index]

        return value
    
    def __eq__(self, other):
        if not isinstance(other, State_key) or self.__hash__() != other.__hash__():
            return False

        return True
        
class Q(QLearning):
    def big_test(self, path = "out.txt"):
        with open(path, "a") as f:
            f.write(str(self.to_i) + "/" + str(self.t + self.base_samples))
            
            clf = RandomForestClassifier(max_depth=self.m_depth, n_jobs=-1)
            clf.fit(self.X_used[:self.to_i], self.y_used[:self.to_i])
            
            predict_arr = clf.predict(self.X_big_test)
            
            f.write(f"q_learning_acc: {accuracy_score(self.y_big_test, predict_arr):.4f}" + "\n")
            
            val = 0
            for _ in range(3):
                clf = RandomForestClassifier(max_depth=self.m_depth, n_jobs=-1)
                indices = np.random.choice(self.base_samples + self.t, self.to_i, replace=False)
        
                clf.fit(self.X[indices], self.y[indices])
                
                predict_arr = clf.predict(self.X_big_test)

                val += accuracy_score(self.y_big_test, predict_arr)

            val /= 3
            f.write(f"random_learning_acc: {val:.4f}" + "\n")
            
            clf = RandomForestClassifier(max_depth=self.m_depth, n_jobs=-1)
            clf.fit(self.X[:self.base_samples + self.t], self.y[:self.base_samples + self.t])
            
            predict_arr = clf.predict(self.X_big_test)
            
            f.write(f"total_learning_acc: {accuracy_score(self.y_big_test, predict_arr):.4f}" + "\n")
            
            q_df = self.q_df
            q_df = q_df.sort_values(by=["q_value"], ascending=False)
            f.write(str(q_df.head()) + "\n\n")

    def get_clf_prediction(self, index, next_class):

        res_index = np.where(self.clf.classes_ == next_class)
        
        proba = self.clf.predict_proba(self.X[index].reshape(1, -1))[0]
        hit = (self.clf.predict(self.X[index].reshape(1, -1)) == self.y[index])[0]

        if len(res_index[0]):
            proba = proba[res_index[0][0]]
        else:
            proba = 0
        
        return (proba, hit)
    
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

    def update_state(self, state_key, action_key):
        sample_index = self.base_i + self.t + 1

        next_class = self.y[sample_index]
        
        if action_key == 1:
            prev_duration = self.duration_into_discrete(self.X_used[self.to_i - 1][94])
            prev_ppi_duration = self.ppi_duration_into_discrete(self.X_used[self.to_i - 1][97])
            prev_class = self.y_used[self.to_i - 1]

            for i in range(self.num_of_sizes):
                self.class_size_sums[prev_class][i] = self.X_used[self.to_i - 1][i + 60]

            self.duration_amount[prev_duration] += 1
            self.ppi_duration_amount[prev_ppi_duration] += 1
            self.class_amount[prev_class] += 1
            self.used += 1

        class_percent = self.class_amount[next_class] / (self.used + self.base_samples)

        client_bytes = self.client_bytes_into_discrete(self.X[sample_index][90])
        server_bytes = self.server_bytes_into_discrete(self.X[sample_index][91])

        duration = self.duration_into_discrete(self.X[sample_index][94])
        percent_duration = self.duration_amount[duration] / (self.used + self.base_samples)

        ppi_duration = self.ppi_duration_into_discrete(self.X[sample_index][97])
        ppi_percent_duration = self.ppi_duration_amount[ppi_duration] / (self.used + self.base_samples)

        centroid_size = self.calculate_centroid_size(next_class, self.X[sample_index])

        return State_key(self.class_percent_into_discrete(class_percent), 
                         duration,
                         self.percent_duration_into_discrete(percent_duration),
                         client_bytes,
                         server_bytes,
                         ppi_duration,
                         self.ppi_percent_duration_into_discrete(ppi_percent_duration),
                         self.centroid_size_into_discrete(centroid_size))

    def initialize(self, cols, iters, already_used, nclasses, epsilon = 0.9, alpha = 0.2, gamma = 0.9):
        self.q_count = defaultdict(int)

        self.epsilon_greedy_rate = epsilon
        self.alpha_value         = alpha
        self.gamma_value         = gamma

        self.m_depth = 15

        if nclasses < 25:
            self.CLASS_PERCENT_VALUES    = [0.01, 0.05, 0.1, 0.2]
        else:
            self.CLASS_PERCENT_VALUES    = [0.001, 0.01, 0.05, 0.1]

        self.PREDICT_PROBA_VALUES        = [0.25, 0.50, 0.75]
        self.DURATION_VALUES             = [0.1, 1, 29.9, 59.9, 89.9, 119.9, 299]
        self.DURATION_PERCENT_VALUES     = [0.05, 0.1, 0.2, 0.4]
        self.PPI_DURATION_VALUES         = [0.2, 9.9, 19.9, 70, 112]
        self.PPI_DURATION_PERCENT_VALUES = [0.005, 0.01, 0.1, 0.4]
        self.CENTROID_SIZE_VALUES        = [300, 500, 800]

        self.used         = 0
        self.base_samples = already_used
        self.base_i       = already_used - 1
        self.to_i         = already_used

        self.class_amount        = defaultdict(int)
        self.duration_amount     = defaultdict(int)
        self.ppi_duration_amount = defaultdict(int)

        self.X_used = np.ndarray(shape = (iters + already_used, cols))
        self.y_used = np.ndarray(shape = (iters + already_used,))
        self.last_f1 = 0

        ## add base samples to state and learn the first iter of classifier
        self.X_used[:self.base_samples] = self.X[:self.base_samples]
        self.y_used[:self.base_samples] = self.y[:self.base_samples]

        self.clf = RandomForestClassifier(max_depth=self.m_depth, n_jobs=-1)
        self.last_f1 = self.test_acc()
        
        self.num_of_sizes = 12

        self.class_size_sums = {}
        for class_i in range(200):
            self.class_size_sums[class_i] = []
            for _ in range(self.num_of_sizes):
                self.class_size_sums[class_i].append(0)

        for i in range(self.base_samples):
            self.class_amount[self.y[i]] += 1
            self.duration_amount[self.duration_into_discrete(self.X[i][94])] += 1
            self.used += 1
    
    def extract_possible_actions(self, state_key):
        return list({0, 1})

    def select_action(self, state_key, next_action_list):
        epsilon_greedy_flag = bool(np.random.binomial(n=1, p=self.epsilon_greedy_rate))

        if epsilon_greedy_flag is False:
            action_key = random.choice(next_action_list)
        else:
            action_key = self.predict_next_action(state_key, next_action_list)

        return action_key

    def train_clf(self, clf):
        clf.fit(self.X_used[:self.to_i], 
                self.y_used[:self.to_i])

    def test_acc(self):
        self.clf = RandomForestClassifier(max_depth=self.m_depth, n_jobs=-1, random_state=10)

        self.train_clf(self.clf)

        predict_arr = self.clf.predict(self.X_test)

        return f1_score(self.y_test, predict_arr, average="weighted")

    def observe_reward_value(self, state_key, action_key):
        pass

    def observe_acc_reward(self, action_key):
        cur_f1 = self.test_acc()
        reward = cur_f1 - self.last_f1

        self.last_f1 = cur_f1

        return reward

    def observe_hit_reward(self, action_key):
        (proba, hit) = self.get_clf_prediction(self.base_i + self.t, self.y[self.base_i + self.t])

        proba_reward = (0.7 - proba)
        hit_reward = -1 if hit == 1 else 1

        if action_key == 1:
            self.X_used[self.to_i] = self.X[self.base_i + self.t]
            self.y_used[self.to_i] = self.y[self.base_i + self.t]
            self.to_i += 1

        return (proba_reward, hit_reward)

    def learn(self, state_key, batch = 1, limit=1000, increased_rd = 1, decrease_alpha = 0):
        self.t = 1
        last_t = 1

        seen_states = []

        for _ in tqdm(range(1, limit + 1)):
            if self.t - last_t > 999:
                self.big_test()
                last_t = self.t

            self.epsilon_greedy_rate = min(self.t / increased_rd, 0.9)
            self.alpha_value = max(self.alpha_value - decrease_alpha, 0.05)
            
            next_action_list = self.extract_possible_actions(state_key)
            action_key = self.select_action(
                state_key=state_key,
                next_action_list=next_action_list
            )

            (proba, hit) = self.observe_hit_reward(action_key)
            # seen_states.append([state_key, reward_value])

            # Max-Q-Value in next action time.
            next_state_key = self.update_state(
                state_key=state_key,
                action_key=action_key
            )

            next_next_action_list = self.extract_possible_actions(next_state_key)
            next_action_key = self.predict_next_action(next_state_key, next_next_action_list)
            next_max_q = self.extract_q_df(next_state_key, next_action_key)

            seen_states.append([state_key, action_key, proba, hit, next_max_q])

            if self.t % batch == 1:
                reward = self.observe_acc_reward(action_key)

                for el in seen_states:
                    (state_key, action_key, proba, hit, next_max_q) = el

                    # Tried with values of 0.98 0.01  0.01
                    #                      0.5  0.3   0.2
                    #                      0.7  0.2   0.1
                    #                      0.9  0.05  0.05
                    #                      0.95 0.025 0.025
                    #
                    # This one worked best 0.97 0.015 0.015
                    # For a subset of 10 classes 0.97 0.015 0.005
                    reward_value = (0.97 * reward + 0.015 * proba + 0.005 * hit)

                    if action_key == 0:
                        reward_value = -reward_value

                    self.update_q(
                        state_key=state_key,
                        action_key=action_key,
                        reward_value=reward_value,
                        next_max_q=next_max_q
                    )

                    self.save_r_df(state_key, reward_value)

                seen_states = []

            # Update State.
            state_key = next_state_key

            # Normalize.
            self.normalize_q_value()
            self.normalize_r_value()

            # # Vis.
            # self.visualize_learning_result(state_key)
            # # Check.
            # if self.check_the_end_flag(state_key) is True:
            #     break

            self.t += 1

    def save_q_df(self, state_key, action_key, q_value):
        if isinstance(q_value, float) is False:
            raise TypeError("The type of q_value must be float.")

        new_q_df = pd.DataFrame([(state_key, action_key, q_value)], columns=["state_key", "action_key", "q_value"])
        
        if q_value != 0.0:
            self.q_count[(state_key, action_key)] += 1

        if self.q_df is not None:
            self.q_df = pd.concat([new_q_df, self.q_df])
            self.q_df = self.q_df.drop_duplicates(["state_key", "action_key"])
        else:
            self.q_df = new_q_df

    def export_table(self):
        return (self.q_df, self.r_df)
    
    def save_table_to_file(self, path = "out_state.txt"):
        with open(path, "w") as f:
            f.write("qvalues\n---------------\n")
            for _, row in self.q_df.iterrows():
                for el in row:
                    if type(el) == State_key:
                        f.write(el.write() + " ")
                    else:
                        f.write(str(el) + " ")

                f.write("\n")

            f.write("rewards\n---------------\n")
            for _, row in self.r_df.iterrows():
                for el in row:
                    if type(el) == State_key:
                        f.write(el.write() + " ")
                    else:
                        f.write(str(el) + " ")

                f.write("\n")
    
    def import_table(self, q_df, r_df):
        self.q_df = q_df
        self.r_df = r_df