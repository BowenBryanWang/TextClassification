import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRNN(nn.Module):
    def __init__(self, args):
        super(TextRNN, self).__init__()
        self.args = args

        class_num = args.class_num  # 分类的类别数
        chanel_num = 1  # 通道数
        hidden_size = 128

        vocabulary_size = args.vocabulary_size  # 已知词的数量
        embedding_dimension = args.embedding_dim  # 每个词向量长度
        self.embedding = nn.Embedding(
            vocabulary_size, embedding_dimension)  # 词向量，这里直接随机
        if args.static:
            self.embedding = self.embedding.from_pretrained(
                args.vectors, freeze=not args.non_static)  # 预先训练好的模型

        if args.multichannel:
            self.embedding2 = nn.Embedding(
                vocabulary_size, embedding_dimension).from_pretrained(args.vectors)  # 多通道的设置
            chanel_num += 1
        else:
            self.embedding2 = None
        self.lstm = nn.LSTM(embedding_dimension, hidden_size,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(hidden_size * 2, class_num)  # 全连接层

    def forward(self, x):
        embed = self.embedding(x)
        lstmout, _ = self.lstm(embed)
        fc_input = lstmout[:, -1, :]
        out = self.fc(fc_input)
        return out
