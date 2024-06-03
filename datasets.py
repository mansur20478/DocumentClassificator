import os
import pathlib
import numpy as np
from sklearn.datasets import fetch_20newsgroups


def load_data(name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, 'data')
    if name == 'fetch_20newsgroups':
        dir_path = os.path.join(dir_path, 'fetch_20newsgroups')
        news = fetch_20newsgroups(data_home=dir_path)
        return news.data, news.target, news.target_names
    elif name == 'bbc':
        dir_path = os.path.join(dir_path, 'bbc')
        folders = os.listdir(dir_path)
        data, target, target_name = [], [], []
        for label in folders:
            data_path = os.path.join(dir_path, label)
            if data_path.endswith("README.TXT"):
                continue
            target_name.append(label)
            label_id = len(target_name) - 1
            for j in os.listdir(data_path):
                new_path = os.path.join(data_path,j)
                text = open(new_path, 'r', encoding='latin-1').read()
                target.append(label_id)
                data.append(text)
        return data, np.array(target), target_name
    else:
        raise ValueError(f'{name} dataset does not exist.')