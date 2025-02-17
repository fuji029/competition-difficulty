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
with open("data/dev/dev.label", "r") as f:
    data = f.read().rstrip().split("\n")
    gold = [list(map(int, item.split("\t"))) for item in data]
    gold = get_gold_labels(pred, [item[0]
                           for item in gold], [item[1] for item in gold])

for p, g in zip(pred, gold):
    if (np.abs(p - g) > 2):
        print(f"{p}, {g}")
print(cohen_kappa_score(gold, pred, weights='quadratic'))
print(classification_report(gold, pred))
