from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
import numpy as np
from pprint import pprint


def get_gold_labels(predictions, lower_labels, higher_labels):
    if np.sum(predictions == lower_labels) >= np.sum(predictions == higher_labels):
        gold_labels = lower_labels
        gold_labels[predictions ==
                    higher_labels] = higher_labels[predictions == higher_labels]
    else:
        gold_labels = higher_labels
        gold_labels[predictions ==
                    lower_labels] = lower_labels[predictions == lower_labels]
    return gold_labels


path = input("path:")
with open(path, "r") as f:
    pred = list(map(int, f.read().rstrip().split("\n")))
    pred = [item for item in pred]
with open("data/dev/dev.label", "r") as f:
    data = f.read().rstrip().split("\n")
    labels = [list(map(int, item.split("\t"))) for item in data]
    gold = get_gold_labels(np.array(pred),
                           np.array([item[0] for item in labels]), np.array([item[1] for item in labels]))


print("QWK=", cohen_kappa_score(gold, pred, weights='quadratic'))
print(classification_report(gold, pred))
