import datetime
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import argparse
import warnings

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-path", required=True)
parser.add_argument("-name", required=True)
args = parser.parse_args()
model_path = args.path
model_name = args.name

tokenizer = AutoTokenizer.from_pretrained(model_name)


def load_label(trg):
    label = []
    with open("data/{}/{}.label".format(trg, trg), "r") as f:
        lines = f.read().rstrip().split("\n")
        for line in lines:
            d1, d2 = map(int, line.split("\t"))
            label.append(d1 - 1)
    return np.array(label, dtype=int)


dev_label = load_label("dev")


def load_text(trg):
    with open("data/{}/{}.txt".format(trg, trg), "r") as f:
        lines = f.read().rstrip().split("\n")
    return np.array(lines, dtype=object)


dev_txt = load_text("test")

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def get_dataset(txts, labels):
    dataset = []
    for txt, label in zip(txts, labels):
        mydict = tokenizer(txt, max_length=32,
                           padding="max_length", truncation=True)
        mydict["labels"] = label
        mydict = {key: torch.tensor(value).to(device)
                  for key, value in mydict.items()}
        dataset.append(mydict)
    return dataset


dev_dataset = get_dataset(dev_txt, dev_label)

dataloader_val = DataLoader(dev_dataset, batch_size=256, shuffle=False)


class Bert4Classification(pl.LightningModule):

    # モデルの読み込みなど。損失関数は自動的に設定される。
    # num_labels == 1 -> 回帰タスクなので MSELoss()
    # num_labels > 1 -> 分類タスクなので CrossEntropyLoss()
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()    # num_labelsとlrを保存する。例えば、self.hparams.lrでlrにアクセスできる。
        self.bert_sc = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)

    # 訓練用データのバッチを受け取って損失を計算
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log("train_loss", loss)
        return loss

    # 検証用データのバッチを受け取って損失を計算
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
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


model = Bert4Classification.load_from_checkpoint(
    model_path, num_labels=6, lr=1e-6)

with torch.no_grad():
    preds = list()
    for batch in dataloader_val:
        output = model.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        preds.append(labels_predicted)
    preds = torch.cat(preds)
    pred_label = [label.item() for label in preds]
    # print(classification_report(dev_label, pred_label))
    # print(cohen_kappa_score(dev_label, pred_label, weights='quadratic'))

now = datetime.datetime.now()
nowtime = now.strftime('%Y%m%d_%H%M%S')
with open(f"output/{nowtime}.txt", "w") as f:
    f.write("".join([f"{pred+1}\n" for pred in pred_label]))
