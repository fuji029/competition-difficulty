# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou
import numpy as np
from eda import *

# arguments to be parsed from command line
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str,
                help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str,
                help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int,
                help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float,
                help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float,
                help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float,
                help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float,
                help="percent of words in each sentence to be deleted")
args = ap.parse_args()

# the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))

# number of augmented sentences to generate per original sentence
num_aug = 9  # default
if args.num_aug:
    num_aug = args.num_aug

# how much to replace each word by synonyms
alpha_sr = 0.1  # default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

# how much to insert new words that are synonyms
alpha_ri = 0.1  # default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

# how much to swap words
alpha_rs = 0.1  # default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

# how much to delete words
alpha_rd = 0.1  # default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
    ap.error('At least one alpha should be greater than zero')

# generate more data with standard augmentation

def load_label(trg, idx):
    labels = []
    with open(trg, "r") as f:
        lines = f.read().rstrip().split("\n")
        for line in lines:
            label = list(map(int, line.split("\t")))
            labels.append(label[idx])
    return np.array(labels, dtype=int)

num_labels = [[], []]
for idx in (0, 1):
    train_label = load_label("data/train/train.label", idx)

    for i in range(6):
        num_labels[idx].append(list(train_label == i+1).count(True))


def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):

    writers_txt = [open(f"{output_file}.0.txt", 'w'),
                   open(f"{output_file}.1.txt", 'w')]
    writers_label = [open(f"{output_file}.0.label", 'w'),
                     open(f"{output_file}.1.label", 'w')]

    src_txt = open(f"{train_orig}.txt", 'r').read().rstrip().split("\n")
    src_label = open(f"{train_orig}.label", 'r').read().rstrip().split("\n")

    for labels, sentence in zip(src_label, src_txt):
        ls = list(map(int, labels.split("\t")))
        for writer_label, writer_txt, l in zip(writers_label, writers_txt, ls):
            if (l in [3, 4]):
                writer_label.write(labels + '\n')
                writer_txt.write(sentence + "\n")
                continue
            elif (l == 1):
                aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri,
                                    alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=29)
            elif (l == 2):
                aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri,
                                    alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=3)
            elif (l == 6):
                aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri,
                                    alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=49)
            else:
                aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri,
                                    alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=3)
            for aug_sentence in aug_sentences:
                writer_label.write(labels + '\n')
                writer_txt.write(aug_sentence + "\n")
    for writer_txt, writer_label in zip(writers_txt, writers_label):
        writer_txt.close()
        writer_label.close()
    print("generated augmented sentences with eda for " + train_orig +
          " to " + output_file + " with num_aug=" + str(num_aug))


# main function
if __name__ == "__main__":

    # generate augmented sentences and output into a new file
    gen_eda(args.input, output, alpha_sr=alpha_sr, alpha_ri=alpha_ri,
            alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
