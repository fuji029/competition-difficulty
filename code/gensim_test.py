import gensim.downloader

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')

print(glove_vectors.most_similar('piezoelectric'))
