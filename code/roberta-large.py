
import torch
import numpy as np
from transformers import RobertaForSequenceClassification
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import RobertaTokenizer

model_name = 'FacebookAI/roberta-large'
tokenizer = RobertaTokenizer.from_pretrained(model_name)


def load_label(trg):
    label = []
    with open("data/{}/{}.label".format(trg, trg), "r") as f:
        lines = f.read().rstrip().split("\n")
        for line in lines:
            d1, d2 = map(int, line.split("\t"))
            label.append(d1 - 1)
    return np.array(label, dtype=int)

train_label = load_label("train")
dev_label = load_label("dev")
test_label = np.zeros(1843, dtype=int)




def load_text(trg):
    with open("data/{}/{}.txt".format(trg, trg), "r") as f:
        lines = f.read().rstrip().split("\n")
    return np.array(lines, dtype=object)

train_txt = load_text("train")
dev_txt = load_text("dev")
test_txt = load_text("test")


def get_dataset(txts, labels):
    dataset = []
    for txt, label in zip(txts, labels):
        mydict = tokenizer(txt, max_length=32, padding="max_length", truncation=True)
        mydict["labels"] = label
        mydict = {key: torch.tensor(value) for key, value in mydict.items()}
        dataset.append(mydict)
    return dataset

train_dataset = get_dataset(train_txt, train_label)
dev_dataset = get_dataset(dev_txt, dev_label)
test_dataset = get_dataset(test_txt, test_label)



dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_val = DataLoader(dev_dataset, batch_size=256, shuffle=False)
dataloader_test = DataLoader(test_dataset, batch_size=256, shuffle=False)


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
        self.bert_sc = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

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


# ====================
# 訓練
# ====================

model = Bert4Classification(model_name, num_labels=6, lr=1e-6)

# 訓練中にモデルを保存するための設定
checkpoint = pl.callbacks.ModelCheckpoint(
    # 検証用データにおける損失が最も小さいモデルを保存する
    monitor="val_loss", mode="min", save_top_k=1,
    # モデルファイル（重みのみ）を "model" というディレクトリに保存する
    save_weights_only=True, dirpath="model/roberta/large/lr1e-6"
)

early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=3)

# 訓練
trainer = pl.Trainer(accelerator="auto", max_epochs=30, callbacks=[checkpoint, early_stopping])
trainer.fit(model, dataloader_train, dataloader_val)

# ベストモデルの確認
print("ベストモデル: ", checkpoint.best_model_path)
print("ベストモデルの検証用データにおける損失: ", checkpoint.best_model_score)


with torch.no_grad():
    preds = list()
    for batch in dataloader_test:
        output = model.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        preds.append(labels_predicted)
    preds = torch.cat(preds)
    pred_label = [label.item() for label in preds]


with open("out.roberta.large.lr1e-6.txt", "w") as f:
    preds = preds.tolist()
    for pred in preds:
        f.write("{}\n".format(pred+1))

from sklearn.metrics import cohen_kappa_score
with torch.no_grad():
    preds = list()
    for batch in dataloader_val:
        output = model.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        preds.append(labels_predicted)
    preds = torch.cat(preds)
    pred_label = [label.item() for label in preds]
print(cohen_kappa_score(dev_label, pred_label, weights='quadratic'))




