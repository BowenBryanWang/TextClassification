import re
from torchtext.legacy import data
import jieba
import logging
jieba.setLogLevel(logging.INFO)

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')


def word_cut(text):
    return text


def get_dataset(path, text_field, label_field):
    text_field.tokenize = word_cut
    train, dev = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train='mytrain.tsv', validation='myvalidation.tsv',
        fields=[
            # ('index', None),
            ('label', label_field),
            ('text', text_field)
        ]
    )
    return train, dev
