import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        class_num = args.class_num  # 分类的类别数
        chanel_num = 1  # 通道数
        filter_num = args.filter_num  # 卷积核的数量
        filter_sizes = args.filter_sizes  # 卷积核的大小

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
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)  # 全连接层

    def forward(self, x):
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
