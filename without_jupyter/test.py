from Qlearning import State_key, Q, used_features
from help import QUIC_dataset, TLS_dataset, Qlearning_tester, create_balanced_test_data

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset',  type=str)
    parser.add_argument('-n', '--nclasses', type=int)
    parser.add_argument('-i', '--iters',    type=int)

    args = parser.parse_args()

    nclasses = 10
    if args.nclasses != None:
        nclasses = args.nclasses

    if args.dataset == None or parser.dataset == "QUIC":
        (train_dataframe, val_dataframe, test_dataframe) = QUIC_dataset(nclasses)
    else:
        (train_dataframe, val_dataframe, test_dataframe) = TLS_dataset(nclasses)

    X = train_dataframe.drop(columns="APP").to_numpy()
    y = train_dataframe["APP"].to_numpy()

    X_test = test_dataframe.drop(columns="APP").to_numpy()[:100000]
    y_test = test_dataframe["APP"].to_numpy()[:100000]

    iters = 700000
    nfeatures = X.shape[1]

    q_tester = Qlearning_tester(iters, nfeatures)

    q_tester.load_q_df("out_state.txt")

    q_tester.X = X
    q_tester.y = y
    q_tester.test(iters)

    q_tester.test_acc(iters, X_test, y_test)
