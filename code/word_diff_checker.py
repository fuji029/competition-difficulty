with open("./data/unified_dataset.csv", "r") as f:
    data = f.read().rstrip().split("\n")[1:]
    csv_data = [row.split(",")[1:] for row in data]
    dl = {"A1":1, "A2":2, "B1":3, "B2":4, "C1":5, "C2":6}
    with open("data/train/train.txt", "r") as f:
        data = f.read().rstrip().split("\n")
    texts = [text.split() for text in data]
    with open("data/train/train.label", "r") as f:
        data = f.read().rstrip().split("\n")
    labels = [list(map(int, text.split("\t"))) for text in data]
    for i in range(len(csv_data)):
        csv_data[i][1] = dl[csv_data[i][1]]
for a in range(1, 7):
    c1 = [item[0] for item in csv_data if item[1] == a]

    data = [[texts[i] , labels[i]] for i in range(len(texts)) if a in labels[i]]

    cnt = 0
    for text, label in data:
        for word in text:
            if word in c1:
                cnt += 1
                break
    print(cnt / len(data))