import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model", required=True)
parser.add_argument("-nowtime", required=True)
parser.add_argument("-gpu", required=True, type=int)
parser.add_argument("-train", default="data/train/train")
parser.add_argument("-dev", default="data/dev/dev")
parser.add_argument("-test", default="data/test/test")
parser.add_argument("-lr", type=float, default=1e-6)

args = parser.parse_args()

nowtime = args.nowtime

model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name)
path = []
path.append(input("path0:"))
path.append(input("path1:"))


def load_label(trg):
    l1 = []
    l2 = []
    with open(f"{trg}.label", "r") as f:
        lines = f.read().rstrip().split("\n")
        for line in lines:
            d1, d2 = map(int, line.split("\t"))
            l1.append(d1 - 1)
            l2.append(d2 - 1)
    return [np.array(l1, dtype=int), np.array(l2, dtype=int)]


dev_labels = load_label(args.dev)


def load_text(trg):
    with open(f"{trg}.txt", "r") as f:
        lines = f.read().rstrip().split("\n")
    return np.array(lines, dtype=object)


dev_txt = load_text(args.dev)


device = torch.device(
    f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print(device)


def get_dataset_eval(txts, labels):
    dataset = []
    for txt, label in zip(txts, labels):
        mydict = tokenizer(txt, max_length=32,
                           padding="max_length", truncation=True)
        mydict["labels"] = label
        mydict = {key: torch.tensor(value).to(device)
                  for key, value in mydict.items()}
        dataset.append(mydict)
    return dataset

# ====================
# BERTによるテキスト分類
# ====================


class LDLLoss(nn.Module):
    """
    Label Distribution Learning Loss
    """

    def __init__(self, dist_size, delta: float = 1e-6):
        """
        損失関数の初期設定

        Args:
            dist_size (int): 確率分布のサイズ
            delta (float): log(0)を防ぐための微少数
        """
        super(LDLLoss, self).__init__()
        self.dist_size = dist_size
        self.delta = delta

    def forward(self, P, y) -> torch.tensor:
        """
        損失の計算

        Args:
            P (torch.tensor(batch_size, self.dist_size)): 予測確率分布
            y (torch.tensor(batch_size)): 正解ラベル
        """
        # 正解クラス y を y を中心とした正規分布 Y に変換
        Y = self.norm_dist(y)
        # Cross Entropy
        loss = -Y * torch.log(P + self.delta)
        return torch.mean(loss)

    def norm_dist(self, y: torch.tensor, sigma: float = 1.0) -> torch.tensor:
        """
        正解ラベルを正規分布に変換する処理

        Args:
            y (torch.tensor(batch_size)): 正規分布の平均
            sigma (float): 正規分布の分散

        Returns:
            torch.tensor(batch_size, self.dist_size): 正規分布
        """
        batch_size = y.size(0)
        X = torch.arange(0, self.dist_size,
                         device=device).repeat(batch_size, 1)
        N = torch.exp(-torch.square(X - y.unsqueeze(1)) / (2 * sigma**2))
        d = torch.sqrt(torch.tensor(2 * np.pi * sigma**2, device=device))

        return N / d


class Bert4Classification(pl.LightningModule):

    # モデルの読み込みなど。損失関数は自動的に設定される。
    # num_labels == 1 -> 回帰タスクなので MSELoss()
    # num_labels > 1 -> 分類タスクなので CrossEntropyLoss()
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()    # num_labelsとlrを保存する。例えば、self.hparams.lrでlrにアクセスできる。
        self.bert_sc = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        self.loss = LDLLoss(num_labels).forward

    # 訓練用データのバッチを受け取って損失を計算
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        softmax = nn.Softmax(dim=1)
        P = softmax(output['logits'])
        y = batch['labels']
        loss = self.loss(P, y)
        # y_hat = output.logits.argmax(-1)
        # y_hat = [label.item() for label in y_hat]
        # loss = torch.Tensor([1 - cohen_kappa_score(y.cpu(), y_hat, weights='quadratic')])
        # loss.requires_grad_(True)
        self.log("train_loss", loss)
        return loss

    # 検証用データのバッチを受け取って損失を計算
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        softmax = nn.Softmax(dim=1)
        P = softmax(output['logits'])
        y = batch['labels']
        val_loss = self.loss(P, y)
        # y_hat = output.logits.argmax(-1)
        # y_hat = [label.item() for label in y_hat]
        # val_loss = torch.Tensor([1 - cohen_kappa_score(y.cpu(), y_hat, weights='quadratic')])
        self.log("val_loss", val_loss)

    # 評価用データのバッチを受け取って分類の正解率を計算
    def test_step(self, batch, batch_idx):
        # ラベルの推定
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        # 正解率の計算
        labels = batch.pop("labels")
        num_correct = (labels_predicted == labels).sum().item()
        accuracy = num_correct / labels.size(0)
        self.log("accuracy", accuracy)

    # 最適化手法を設定
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ====================
# 訓練
# ====================
dev_preds = []
test_preds = []

for i, dev_label in enumerate(dev_labels):
    dev_dataset = get_dataset_eval(dev_txt, dev_label)
    dataloader_val = DataLoader(dev_dataset, batch_size=256, shuffle=False)

    model = Bert4Classification(model_name, num_labels=6, lr=args.lr)
    model = model.load_from_checkpoint(
        path[i], num_labels=6, lr=1e-6, map_location=device)

    with torch.no_grad():
        preds = list()
        logits = list()
        for batch in dataloader_val:
            output = model.bert_sc(**batch)
            labels_predicted = output.logits.argmax(-1)
            preds.append(labels_predicted)
            for idx, logit in zip(labels_predicted, output.logits):
                logits.append(logit[idx].item())
        preds = torch.cat(preds)
        pred_label = [[label.item(), logit]
                      for label, logit in zip(preds, logits)]
        dev_preds.append(pred_label)


def logits_ensembling(preds):
    ps1 = preds[0]
    ps2 = preds[1]
    labels = []
    for p1, p2 in zip(ps1, ps2):
        labels.append(p1[0] if p1[1] > p2[1] else p2[0])
    return labels


dev_preds = logits_ensembling(dev_preds)

with open(f"output/dev/{nowtime}.txt", "w") as f:
    for pred in dev_preds:
        f.write("{}\n".format(pred+1))
