# Evaluate model performance

import pickle
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


def eval_model():
    # Loading test data
    print("Loading data and model...")
    test_data = np.load('./data/processed_test_data.npy')

    # Loading trained model
    with open('./data/model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("done.")

    # Dividing loaded data-set into data and labels
    labels = test_data[:, 0]
    data = test_data[:, 1:]

    # Running model on test data
    print("Running model on test data...")
    predictions = model.predict(data)
    print("done.")

    # Calculating metric scores
    print("Calculating metrics...")
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')

    metrics = {'accuracy': accuracy, 'f1_score': f1, 'recall': recall}

    conf_mat_cnn = confusion_matrix(labels, predictions)
    f,ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_mat_cnn, annot=True, linewidths=0.01,cmap="cubehelix",linecolor="gray", fmt= '.1f',ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('metrics/confmat.png', dpi=120)
    # Saving metrics to txt file for cml
    with open("metrics/metrics.txt", 'w') as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"f1 Score: {f1}\n")
        file.write(f"Recall Score: {recall}\n")

    # Saving metrics to json file
    json_object = json.dumps(metrics, indent=4)
    with open('./metrics/eval.json', 'w') as f:
        f.write(json_object)
    print("done.")


if __name__ == '__main__':
    eval_model()