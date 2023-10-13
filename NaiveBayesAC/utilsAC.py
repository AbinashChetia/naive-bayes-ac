import numpy as np
import pandas as pd
import re

def text2df(text):
    if isinstance(text, dict):
        for k in list(text):
            k_new = re.sub(r'[^\w\s]', '', k.lower())
            text[k_new] = text.pop(k)
        words_per_data = [[word for word in k.split()] for k in text.keys()]
        uniq_words = list(set([word for sublist in words_per_data for word in sublist]))
        data = pd.DataFrame(np.zeros((len(text), len(uniq_words) + 1)), columns=uniq_words + ['y'])
        idx = 0
        for k in text.keys():
            for word in k.split():
                data.loc[idx, word] += 1
            if text[k] == 'spam':
                data.loc[idx, 'y'] = 1
            elif text[k] == 'ham':
                data.loc[idx, 'y'] = 0
            idx += 1
        return data
    elif isinstance(text, list):
        for i in range(len(text)):
            text[i] = re.sub(r'[^\w\s]', '', text[i].lower())
        words_per_data = [[word for word in k.split()] for k in text]
        uniq_words = list(set([word for sublist in words_per_data for word in sublist]))
        data = pd.DataFrame(np.zeros((len(text), len(uniq_words))), columns=uniq_words)
        idx = 0
        for k in text:
            for word in k.split():
                data.loc[idx, word] += 1
            idx += 1
        return data
    else:
        raise ValueError('Labelled text must be a dictionary and unlabelled text must be a list.')

def splitTrainTest(x, y, train_ratio=0.8):
    '''
    Split data into training and testing sets.
    '''
    df_x = x.copy()
    df_y = y.copy()
    df_y = df_y.rename('y')
    df = pd.concat([df_x, df_y], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    train_size = int(len(df) * train_ratio)
    train_x = df.iloc[:train_size, :-1].reset_index(drop=True)
    train_y = df.iloc[:train_size, -1].reset_index(drop=True)
    test_x = df.iloc[train_size:, :-1].reset_index(drop=True)
    test_y = df.iloc[train_size:, -1].reset_index(drop=True)
    return train_x, train_y, test_x, test_y

def split_kfold(x, y, k=5):
    '''
    Split data into training and testing sets for k-fold cross validation.
    '''
    df_x = x.copy()
    df_y = y.copy()
    df_y = df_y.rename('y')
    df = pd.concat([df_x, df_y], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    fold_size = int(len(df) / k)
    data_folds = []
    for i in range(k):
        if i != k - 1:
            data_folds.append(df.iloc[i * fold_size: (i + 1) * fold_size, :].reset_index(drop=True))
        else:
            data_folds.append(df.iloc[i * fold_size:, :].reset_index(drop=True))
    return data_folds
    

def normMinMax(df, mode='train', train_min=None, train_max=None):
    '''
    Perform min-max normalization on data.
    '''
    data = df.copy()
    if mode == 'train':
        train_max = {}
        train_min = {}
        for col in data.columns:
            train_max[col] = data[col].max()
            train_min[col] = data[col].min()
            data[col] = (data[col] - train_min[col]) / (train_max[col] - train_min[col])
        return data, train_min, train_max
    
    elif mode == 'test':
        if train_min is None or train_max is None:
            raise Exception('Pass train_min and/or train_max.')
        for col in data.columns:
            data[col] = (data[col] - train_min[col]) / (train_max[col] - train_min[col])
        return data
    
def get_acc(y, pred):
    acc = 0
    for i in range(len(pred)):
        if pred[i] == y[i]:
            acc += 1
    return acc / len(pred)