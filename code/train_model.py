# Train classification model for MNIST

import json
import pickle
import numpy as np
from sklearn.svm import SVC 
from sklearn.multiclass import OneVsRestClassifier
import time

def train_model():
    # Measure training time
    start_time = time.time()

    # Loading training data
    print("Load training data...")
    train_data = np.load('./data/processed_train_data.npy')

    # REMOVE FROM HERE

    # Choosing a random sample of images from the training data.
    # This is important since SVM training time increases quadratically with the number of training samples.
    print("Choosing smaller sample to shorten training time...")
    # Setting a random seed so that we get the same "random" choices when we try to recreate the experiment.
    np.random.seed(42)

    num_samples = 5000
    choice = np.random.choice(train_data.shape[0], num_samples, replace=False)
    train_data = train_data[choice, :]

    # TILL HERE

    # Dividing loaded data-set into data and labels
    labels = train_data[:, 0]
    data = train_data[:, 1:]
    print("done.")

    # Defining SVM classifier and train model
    print("Training model...")
    model = OneVsRestClassifier(SVC(kernel='linear'), n_jobs=6)
    model.fit(data, labels)
    print("done.")

    # Saving model as pkl
    print("Save model and training time metric...")
    with open("./data/model.pkl", 'wb') as f:
        pickle.dump(model, f)

    # End training time measurement
    end_time = time.time()

    # Creating metric for model training time
    with open('./metrics/train_metric.json', 'w') as f:
        json.dump({'training_time': end_time - start_time}, f)
    print("done.")


if __name__ == '__main__':
    train_model()