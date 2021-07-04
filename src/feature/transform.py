def one_hot_encode(data, feature):
    encode = lambda gender: 1 if gender == 'female' else 0
    data.loc[:, feature] = data[feature].apply(encode)
    return data


def train_test_split(data, label):
    data_train = data[:int(data.shape[0] * 0.8)]
    data_test = data[int(data.shape[0] * 0.8):]
    label_train = label[:int(label.shape[0] * 0.8)]
    label_test = label[int(label.shape[0] * 0.8):]

    assert data_train.shape[0] == label_train.shape[0]
    assert data_test.shape[0] == label_test.shape[0]

    return data_train, data_test, label_train, label_test


