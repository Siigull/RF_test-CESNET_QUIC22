from Qlearning import State_key, Q, used_features
from help import QUIC_dataset, TLS_dataset, Qlearning_tester, create_balanced_test_data

import argparse
from datetime import now

if __name__ == '__main__':
    ##### qlearning default params ######
    increased_rd = 500 # increased randomness for n iters
    decrease_alpha = 0.0001
    iters = 100000
    base_samples_amount = 400
    epsilon = 0.9
    alpha = 0.2
    gamma = 0.95
    nclasses = 10
    batches = 100
    #####################################

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset',  type=str)
    parser.add_argument('-n', '--nclasses', type=int)
    parser.add_argument('-a', '--alpha',    type=float)
    parser.add_argument('-g', '--gamma',    type=float)
    parser.add_argument('-e', '--epsilon',  type=float)
    parser.add_argument('-i', '--iters',    type=int)
    parser.add_argument('-f', '--features', type=str)
    parser.add_argument('-b', '--batches',  type=int)

    args = parser.parse_args()

    if args.nclasses != None:
        nclasses = args.nclasses

    if args.epsilon != None:
        epsilon = args.epsilon

    if args.alpha != None:
        alpha = args.alpha

    if args.gamma != None:
        gamma = args.gamma

    if args.iters != None:
        iters = args.iters

    if args.batches != None:
        batches = args.batches

    if args.features != None:
        features = [int(item) for item in args.features.split(',')]
        if len(features) != len(State_key.__dataclass_fields__):
            print("Wrong number of features")

            exit()
        
        used_features[:] = features

    if args.dataset == None or parser.dataset == "QUIC":
        (train_dataframe, val_dataframe, test_dataframe) = QUIC_dataset(nclasses)
    else:
        (train_dataframe, val_dataframe, test_dataframe) = TLS_dataset(nclasses)

    q = Q()

    q.X = train_dataframe.drop(columns="APP").to_numpy()
    q.y = train_dataframe["APP"].to_numpy()

    q.X_big_test = test_dataframe.drop(columns="APP").to_numpy()[:100000]
    q.y_big_test = test_dataframe["APP"].to_numpy()[:100000]

    nfeatures = q.X.shape[1]

    (q.X_test, q.y_test) = create_balanced_test_data(nfeatures, test_dataframe, nfrom_class = 1000)

    q.initialize(nfeatures, iters, base_samples_amount, epsilon, alpha, gamma)

    state_key = q.update_state(State_key(0, 0, 0, 0, 0, 0, 0, 0), 0)
    q.learn(state_key, batches, iters, increased_rd)

    # time = str(now().strftime('%Y-%m-%d%H:%M:%S'))

    q.big_test("out_acc[batches=" + str(batches) + "].txt")

    q.save_table_to_file("out_state[batches=" + str(batches) + "].txt")