def argmax(l):
    M = max(l)
    for i in range(len(l)):
        if l[i] == M:
            return i


path = input()
predicts = []
while (path != "-1"):
    with open(path, "r") as f:
        l = list(map(int, f.read().rstrip().split("\n")))
        predicts.append(l)
    path = input()
# path = input()
# with open(path, "r") as f:
#     last = list(map(int, f.read().rstrip().split("\n")))
predict = []
for i in range(len(predicts[0])):
    l = [0 for _ in range(6)]
    for j in range(len(predicts)):
        l[predicts[j][i] - 1] += 1
    # predict.append(last[i]-1 if last[i] == 6 else argmax(l))
    predict.append(argmax(l))
for item in predict:
    print(item + 1)
