# Create feature CSVs for train and test datasets

import json 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle
import base64

def featurization():
    # Loading data-sets
    print("Loading data sets...")
    train_data = pd.read_csv('./data/train_data.csv', header=None, dtype=float).values
    test_data = pd.read_csv('./data/test_data.csv', header=None, dtype=float).values
    print("done.")

    # Create PCA object of the 20 most important components
    print("Creating PCA object...")
    pca = PCA(n_components=20, whiten=True)
    pca.fit(train_data[:, 1:])
    train_labels = train_data[:, 0].reshape([train_data.shape[0], 1])
    test_labels = test_data[:, 0].reshape([test_data.shape[0], 1])
    train_data = np.concatenate([train_labels, pca.transform(train_data[:, 1:])], axis=1)
    test_data = np.concatenate([test_labels, pca.transform(test_data[:, 1:])], axis=1)
    print("done.")

    print("Saving processed datasets and PCA params...")
    np.save('./data/processed_train_data.npy', train_data)
    np.save('./data/processed_test_data.npy', test_data)

    with open('./data/norm_params.json', 'w') as f:
        pca_as_string = base64.encodebytes(pickle.dumps(pca)).decode("utf-8")
        json.dump({'pca': pca_as_string }, f)
    print("done.")


if __name__ == '__main__':
    featurization()