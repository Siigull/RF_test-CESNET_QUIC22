from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from pyqlearning.q_learning import QLearning

from collections import defaultdict
from dataclasses import dataclass
from math import log
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

@dataclass
class State_key:
    percent_of_class : int  # 1e-2, 1e-1. 5e-1, 1, more than 1
    predict_proba    : int  # 0-25, 25-50, 50-75, 75-100
    correct_predict  : bool # True or False
    percent_duration : int
    # bytes_client     : int  # log3, 9 buckets
    # bytes_server     : int  # log5, 7 buckets
    duration         : int

    def __hash__(self):
        return self.percent_of_class + \
               self.predict_proba * 5 + \
               int(self.correct_predict) * 20 + \
               self.percent_duration * 40 + \
               self.duration * 320
            #    self.bytes_client * 40 + \
            #    self.bytes_server * 360 + \
               
    
    def __eq__(self, other):
        if not isinstance(other, State_key) or self.__hash__() != other.__hash__():
            return False
        
        return True
        # return self.percent_of_class == other.percent_of_class and \
        #        self.predict_proba == other.predict_proba and \
        #        self.percent_used == other.percent_used and \
        #        self.correct_predict == other.correct_predict
        
class Q(QLearning):
    def big_test(self):
        with open("out.txt", "a") as f:
            f.write(str(self.to_i) + "\n")
            
            clf = RandomForestClassifier()
            clf.fit(self.X_used[:self.to_i], self.y_used[:self.to_i])
            
            predict_arr = clf.predict(self.X_big_test)
            
            f.write(f"q_learning_acc: {accuracy_score(self.y_big_test, predict_arr):.4f}" + "\n")
            

            val = 0
            for i in range(3):
                clf = RandomForestClassifier()
                indices = np.random.choice(self.base_samples + self.t, self.to_i, replace=False)
        
                clf.fit(self.X[indices], self.y[indices])
                
                predict_arr = clf.predict(self.X_big_test)

                val += accuracy_score(self.y_big_test, predict_arr)

            val /= 3
            f.write(f"random_learning_acc: {val:.4f}" + "\n")
            

            clf = RandomForestClassifier()
            clf.fit(self.X[:self.base_samples + self.t], self.y[:self.base_samples + self.t])
            
            predict_arr = clf.predict(self.X_big_test)
            
            f.write(f"total_learning_acc: {accuracy_score(self.y_big_test, predict_arr):.4f}" + "\n")
            
            q_df = self.q_df
            q_df = q_df.sort_values(by=["q_value"], ascending=False)
            f.write(str(q_df.head()) + "\n\n")

    def get_clf_prediction(self, index):

        proba = self.clf.predict_proba(self.X[index].reshape(1, -1))[0]
        hit = (self.clf.predict(self.X[index].reshape(1, -1)) == self.y[index])[0]

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
            
    def client_bytes_into_discrete(self, nbytes):
        return int(np.clip(int(log(nbytes, 3)) - 5, 1, 9) - 1)
    
    def server_bytes_into_discrete(self, nbytes):
        return int(np.clip(int(log(nbytes, 5)) - 4, 0, 7))

    def update_state(self, state_key, action_key, offset=0):
        sample_index = self.base_i + self.t + 1 + offset

        next_class = self.y[sample_index]

        if action_key == 1 and offset == 0:
            prev_duration = self.duration_into_discrete(self.X_used[self.to_i - 1][94])

            self.duration_amount[prev_duration] += 1
            self.class_amount[self.y_used[self.to_i - 1]] += 1
            self.used += 1

        class_percent = self.class_amount[next_class] / (self.used + self.base_samples)
        
        (proba, hit) = self.get_clf_prediction(sample_index)

        res_index = np.where(self.clf.classes_ == next_class)
        if len(res_index[0]):
            proba = self.predict_proba_into_discrete(proba[res_index[0][0]])
        else:
            proba = 0

        # client_bytes = self.client_bytes_into_discrete(X[sample_index][90])
        # server_bytes = self.server_bytes_into_discrete(X[sample_index][91])

        duration = self.duration_into_discrete(self.X[sample_index][94])
        percent_duration = self.duration_amount[duration] / (self.used + self.base_samples)

        return State_key(self.class_percent_into_discrete(class_percent), 
                         proba,
                         hit,
                         self.percent_duration_into_discrete(percent_duration),
                         duration)

    def initialize(self, cols, iters, already_used, epsilon = 0.9, alpha = 0.2, gamma = 0.9):
        self.q_count = defaultdict(int)

        self.epsilon_greedy_rate = epsilon
        self.alpha_value         = alpha
        self.gamma_value         = gamma

        self.CLASS_PERCENT_VALUES    = [0.0001, 0.01, 0.05, 0.1]
        self.PREDICT_PROBA_VALUES    = [0.25, 0.50, 0.75]
        self.DURATION_VALUES         = [0.1, 1, 29.9, 59.9, 89.9, 119.9, 299]
        self.DURATION_PERCENT_VALUES = [0.05, 0.1, 0.2, 0.4]

        self.used         = 0
        self.base_samples = already_used
        self.base_i       = already_used - 1
        self.to_i         = already_used

        self.class_amount    = defaultdict(int)
        self.duration_amount = defaultdict(int)

        self.X_used = np.ndarray(shape = (iters + already_used, cols))
        self.y_used = np.ndarray(shape = (iters + already_used,))
        self.last_f1 = 0

        ## add base samples to state and learn the first iter of classifier
        self.X_used[:self.base_samples] = self.X[:self.base_samples]
        self.y_used[:self.base_samples] = self.y[:self.base_samples]

        self.clf = RandomForestClassifier(max_depth=10, n_jobs=1)
        self.last_f1 = self.test_acc()
        
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
        self.clf = RandomForestClassifier(max_depth=10, n_jobs=1)

        self.train_clf(self.clf)

        predict_arr = self.clf.predict(self.X_test)

        return f1_score(self.y_test, predict_arr, average="weighted")

    def observe_reward_value(self, state_key, action_key):
        self.X_used[self.to_i] = self.X[self.base_i + self.t]
        self.y_used[self.to_i] = self.y[self.base_i + self.t]

        self.to_i += 1

        cur_f1 = self.test_acc()
        reward = cur_f1 - self.last_f1

        if action_key == 0:
            self.to_i -= 1
            reward = -reward
        else:
            self.last_f1 = cur_f1

        self.save_r_df(state_key, reward)

        return reward

    def learn(self, state_key, limit=1000, increased_rd = 1, decrease_alpha = 0):
        self.t = 1
        last_t = 1

        for _ in tqdm(range(1, limit + 1)):
            if self.t - last_t > 1000:
                self.big_test()
                last_t = self.t

            self.epsilon_greedy_rate = min(self.t / increased_rd, 0.9)
            self.alpha_value = max(self.alpha_value - decrease_alpha, 0.05)
            
            next_action_list = self.extract_possible_actions(state_key)
            action_key = self.select_action(
                state_key=state_key,
                next_action_list=next_action_list
            )
            reward_value = self.observe_reward_value(state_key, action_key)

            # Max-Q-Value in next action time.
            next_state_key = self.update_state(
                state_key=state_key,
                action_key=action_key
            )

            next_next_action_list = self.extract_possible_actions(next_state_key)
            next_action_key = self.predict_next_action(next_state_key, next_next_action_list)
            next_max_q = self.extract_q_df(next_state_key, next_action_key)

            # Update Q-Value.
            self.update_q(
                state_key=state_key,
                action_key=action_key,
                reward_value=reward_value,
                next_max_q=next_max_q
            )
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
            for index, row in self.q_df.iterrows():
                for el in row:
                    f.write(str(el) + " ")
                f.write("\n")

            f.write("rewards\n---------------\n")
            for index, row in self.r_df.iterrows():
                for el in row:
                    f.write(str(el) + " ")
                f.write("\n")
    
    def import_table(self, q_df, r_df):
        self.q_df = q_df
        self.r_df = r_df