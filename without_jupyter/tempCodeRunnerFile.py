if __name__ == '__main__':
    (train_dataframe, val_dataframe, test_dataframe) = QUIC_dataset(10)

    X = train_dataframe.drop(columns="APP").to_numpy()
    y = train_dataframe["APP"].to_numpy()

    X_test = test_dataframe.drop(columns="APP").to_numpy()[:100000]
    y_test = test_dataframe["APP"].to_numpy()[:100000]

    iters = 100000
    nfeatures = X.shape[1]

    q_tester = Qlearning_tester(iters, nfeatures)

    q_tester.load_q_df("out_state.txt")

    q_tester.X = X
    q_tester.y = y
    q_tester.test(iters)

    q_tester.test_acc(iters, X_test, y_test)