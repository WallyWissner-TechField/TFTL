import numpy as np


def split(df, fracs):
    """
    Randomly split the dataframe into pieces with size equal to the appropriate fraction of the data specified.
    :param df: Dataframe to split.
    :param fracs: List of fractions for each split. Must sum to 1.
    :param seed: Seed for random split.
    :return: Tuple of dataframes.
    """
    if not np.isclose(sum(fracs), 1):
        raise ValueError("Fractional dataframes must sum to 1.")
    n = df.shape[0]
    # Shuffle the slice of the dataframe.
    shuffled = list(range(n))
    np.random.shuffle(shuffled)
    # Buffer fractions for slicing.
    fracs = [0] + fracs
    # Cumulative fractions.
    cum_fracs = [sum(fracs[:i]) for i in range(1, len(fracs)+1)]
    dfs = tuple(df.iloc[shuffled[int(n*sfrac1): int(n*sfrac2)]] for sfrac1, sfrac2 in zip(cum_fracs, cum_fracs[1:]))
    return dfs


def k_fold():
    pass


def train_validate_test(X, Y, model, train=.6, validate=.2, test=.2, seed=1, **kwargs):
    # Partition the data into train/validate/test.
    np.random.seed(seed)

    n = X.shape[0]
    shuffle = list(range(n))
    np.random.shuffle(shuffle)

    ## Delimiting indices for the partition.
    indices = n * [train, train + validate, 1]
    indices = [int(i) for i in indices]

    ranges = (shuffle[:indices[0]], shuffle[indices[0]: indices[1]], shuffle[indices[1]:])

    def _shuffled(M):
        """
        :param M: np.array
        :return: (M_train, M_validate, M_test)
        """
        return M[ranges[0]], M[ranges[1]], M[ranges[2]]

    X = _shuffled(X)
    Y = _shuffled(Y)

    m = model(*kwargs)
    m.fit(X[0], Y[0])
    print(m.r_squared())


def grid_search(**kwargs):
    """
    :param kwargs:
    :return:
    """

    pass
