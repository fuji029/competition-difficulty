import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument("-")


def load_label(trg, idx):
    labels = []
    with open(trg, "r") as f:
        lines = f.read().rstrip().split("\n")
        for line in lines:
            label = list(map(int, line.split("\t")))
            labels.append(label[idx])
    return np.array(labels, dtype=int)


for idx in (0, 1):
    train_label = load_label("data/train/train.label", idx)
    for i in range(6):
        print(
            f"label {i+1}: {list(train_label == i+1).count(True)}")

# def load_text(trg):
#     with open("data/{}/{}.txt".format(trg, trg), "r") as f:
#         lines = f.read().rstrip().split("\n")
#     return np.array(lines, dtype=object)

# train_txt = load_text("train")
