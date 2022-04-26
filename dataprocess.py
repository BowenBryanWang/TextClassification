import csv
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


def w2v():
    model = KeyedVectors.load_word2vec_format(
        'data/wiki_word2vec_50.bin', binary=True)
    model.save_word2vec_format('data/wiki_word2vec_50.txt', binary=False)


def data_processing(filename):
    # Load data
    with open('data/'+filename+'.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        all = []
        for line in lines:
            words = line.split()
            temp = []
            temp.append(words[0])
            temp.append(words[1:])
            print(temp)
            all.append(temp)
        print(len(all))
        with open(r'data/my'+filename+'.tsv', 'w', newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow(['label', 'text'])
            tsv_w.writerows(np.array(all).tolist())  # 多行写入


# data_processing('train')
# data_processing('test')
# data_processing('validation')
w2v()