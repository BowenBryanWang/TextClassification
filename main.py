import argparse
import torch
import torchtext.legacy.data as data
from torchtext.vocab import Vectors

import TextCNN
import train
import dataset
import TextRNN

parser = argparse.ArgumentParser(description='TextCNN text classifier')
# learning
parser.add_argument('-lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256,
                    help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=128,
                    help='batch size for training [default: 128]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-dir', type=str, default='snapshot',
                    help='where to save the snapshot')
parser.add_argument('-early-stopping', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True,
                    help='whether to save when get best performance')
# model
parser.add_argument('-dropout', type=float, default=0.5,
                    help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0,
                    help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embedding-dim', type=int, default=128,
                    help='number of embedding dimension [default: 128]')
parser.add_argument('-filter-num', type=int, default=100,
                    help='number of each size of filter')
parser.add_argument('-filter-sizes', type=str, default='3,4,5',
                    help='comma-separated filter sizes to use for convolution')

parser.add_argument('-static', type=bool, default=False,
                    help='whether to use static pre-trained word vectors')
parser.add_argument('-non-static', type=bool, default=False,
                    help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('-multichannel', type=bool, default=False,
                    help='whether to use 2 channel of word vectors')
parser.add_argument('-pretrained-name', type=str, default='sgns.zhihu.word',
                    help='filename of pre-trained word vectors')
parser.add_argument('-pretrained-path', type=str,
                    default='pretrained', help='path of pre-trained word vectors')

# device
parser.add_argument('-device', type=int, default=-1,
                    help='device to use for iterate data, -1 mean cpu [default: -1]')

# option
parser.add_argument('-snapshot', type=str, default=None,
                    help='filename of model snapshot [default: None]')
args = parser.parse_args()


def load_word_vectors(model_name, model_path):
    return Vectors(name=model_name, cache=model_path)


def load_dataset(text_field, label_field, args, **kwargs):
    train_dataset, dev_dataset = dataset.get_dataset(
        'data', text_field, label_field)  # 对于数据的处理，这里的数据是从data文件夹中读取的
    # if args.static and args.pretrained_name and args.pretrained_path:
    vectors = load_word_vectors('wiki_word2vec_50.txt', 'data')  # 加载预训练的词向量
    text_field.build_vocab(train_dataset, dev_dataset,
                           vectors=vectors)  # 根据预训练好的词向量，构建词典
    # else:
    # text_field.build_vocab(train_dataset, dev_dataset)
    label_field.build_vocab(train_dataset, dev_dataset)  # 构建标签词典
    train_iter, dev_iter = data.Iterator.splits(
        (train_dataset, dev_dataset),
        batch_sizes=(args.batch_size, len(dev_dataset)),
        sort_key=lambda x: len(x.text),
        **kwargs)  # 构造迭代器
    return train_iter, dev_iter


print('Loading data...')
text_field = data.Field(lower=True)  # 文本域
label_field = data.Field(sequential=False)  # 标签域
train_iter, dev_iter = load_dataset(
    text_field, label_field, args, device=-1, repeat=False, shuffle=True)  # 加载数据集

args.vocabulary_size = len(text_field.vocab)
print('Vocabulary Size: {}'.format(args.vocabulary_size))
if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True
args.class_num = len(label_field.vocab)
print('Classes: {}'.format(args.class_num))
args.cuda = args.device != -1 and torch.cuda.is_available()
print('GPU: {}'.format(args.cuda))
args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]
print('Filter_sizes: {}'.format(args.filter_sizes))

print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))

text_cnn = TextRNN.TextRNN(args)
if args.snapshot:
    print('\nLoading model from {}...\n'.format(args.snapshot))
    text_cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    text_cnn = text_cnn.cuda()
try:
    train.train(train_iter, dev_iter, text_cnn, args)
except KeyboardInterrupt:
    print('Exiting from training early')
