import torch
import numpy as np
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import BertTokenizer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model", required=True)
parser.add_argument("-memo", required=True)
parser.add_argument("-gpu", required=True, type=int)
parser.add_argument("-train", default="data/train/train")
parser.add_argument("-dev", default="data/dev/dev")
parser.add_argument("-test", default="data/test/test")
parser.add_argument("-lr", type=float, default=1e-6)

args = parser.parse_args()

now = datetime.datetime.now()
nowtime = now.strftime('%Y%m%d_%H%M%S')

model_name = args.model
tokenizer = BertTokenizer.from_pretrained(model_name)


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


train_labels = load_label(args.train)
dev_labels = load_label(args.dev)
test_label = np.zeros(1843, dtype=int)


def load_text(trg):
    with open(f"{trg}.txt", "r") as f:
        lines = f.read().rstrip().split("\n")
    return np.array(lines, dtype=object)


train_txt = load_text(args.train)
dev_txt = load_text(args.dev)
test_txt = load_text(args.test)


def get_dataset(txts, labels):
    dataset = []
    for txt, label in zip(txts, labels):
        mydict = tokenizer(txt, max_length=64,
                           padding="max_length", truncation=True)
        mydict["labels"] = label
        mydict = {key: torch.tensor(value) for key, value in mydict.items()}
        dataset.append(mydict)
    return dataset


device = torch.device(
    f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


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


class Bert4Classification(pl.LightningModule):

    # モデルの読み込みなど。損失関数は自動的に設定される。
    # num_labels == 1 -> 回帰タスクなので MSELoss()
    # num_labels > 1 -> 分類タスクなので CrossEntropyLoss()
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()    # num_labelsとlrを保存する。例えば、self.hparams.lrでlrにアクセスできる。
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)

    # 訓練用データのバッチを受け取って損失を計算
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        # y_hat = output.logits.argmax(-1)
        # y_hat = [label.item() for label in y_hat]
        # loss = torch.Tensor([1 - cohen_kappa_score(y.cpu(), y_hat, weights='quadratic')])
        # loss.requires_grad_(True)
        self.log("train_loss", loss)
        return loss

    # 検証用データのバッチを受け取って損失を計算
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
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

for i, (train_label, dev_label) in enumerate(zip(train_labels, dev_labels)):
    train_dataset = get_dataset(train_txt, train_label)
    dev_dataset = get_dataset_eval(dev_txt, dev_label)
    test_dataset = get_dataset_eval(test_txt, test_label)

    dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(dev_dataset, batch_size=256, shuffle=False)
    dataloader_test = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = Bert4Classification(model_name, num_labels=6, lr=args.lr)

    # 訓練中にモデルを保存するための設定
    checkpoint = pl.callbacks.ModelCheckpoint(
        # 検証用データにおける損失が最も小さいモデルを保存する
        monitor="val_loss", mode="min", save_top_k=1,
        # モデルファイル（重みのみ）を "model" というディレクトリに保存する
        save_weights_only=True, dirpath=f"model/{nowtime}/{i}"
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=3)

    # 訓練
    trainer = pl.Trainer(accelerator="auto", max_epochs=30, callbacks=[
        checkpoint, early_stopping], devices=[args.gpu])
    trainer.fit(model, dataloader_train, dataloader_val)

    # ベストモデルの確認
    print("ベストモデル: ", checkpoint.best_model_path)
    print("ベストモデルの検証用データにおける損失: ", checkpoint.best_model_score)
    model = Bert4Classification.load_from_checkpoint(
        checkpoint.best_model_path, num_labels=6, lr=1e-6)
    with torch.no_grad():
        preds = list()
        logits = list()
        for batch in dataloader_test:
            output = model.bert_sc(**batch)
            labels_predicted = output.logits.argmax(-1)
            preds.append(labels_predicted)
            for idx, logit in zip(labels_predicted, output.logits):
                logits.append(logit[idx].item())
        preds = torch.cat(preds)
        pred_label = [[label.item(), logit]
                      for label, logit in zip(preds, logits)]
        test_preds.append(pred_label)

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
test_preds = logits_ensembling(test_preds)


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


dev_gold = get_gold_labels(dev_preds, dev_labels[0], dev_labels[1])

with open(f"output/{nowtime}.txt", "w") as f:
    for pred in test_preds:
        f.write("{}\n".format(pred+1))


print("QCK=", cohen_kappa_score(dev_gold, dev_preds, weights='quadratic'))

print(classification_report(dev_gold, dev_preds))

with open(f"logs/{nowtime}.txt", "w") as f:
    f.write(f"description: {args.memo}\n")
    f.write(f"model: {args.model}\n")
    f.write(f"\tlr: {args.lr}\n")
    f.write(f"ベストモデル: {checkpoint.best_model_path}\n")
    f.write(f"ベストモデルの検証用データにおける損失: {checkpoint.best_model_score}\n")
    f.write(
        f"QCK={cohen_kappa_score(dev_gold, dev_preds, weights='quadratic')}\n")
    f.write(f"{classification_report(dev_gold, dev_preds)}\n")
