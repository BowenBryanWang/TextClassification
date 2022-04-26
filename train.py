import os
import sys
import torch
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 优化器
    steps = 0  # 记录迭代次数
    best_acc = 0  # 记录最佳验证集准确率
    last_step = 0  # 记录上一次保存的训练模型的轮数
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:  # 训练集迭代器，每次小批量训练
            feature, target = batch.text, batch.label  # 获取特征feature和目标target
            with torch.no_grad():
                feature.t_(), target.sub_(1)  # 将target转换为one-hot编码
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()  # 梯度清零
            logits = model(feature)  # 计算模型的输出
            loss = F.cross_entropy(logits, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            steps += 1  # 记录迭代次数
            if steps % args.log_interval == 0:  # 每隔一定的轮数，打印一次训练信息
                corrects = (torch.max(logits, 1)[1].view(
                    target.size()).data == target.data).sum()  # 计算准确率
                train_acc = 100.0 * corrects / batch.batch_size  # 计算训练集准确率
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))  # 打印训练信息
            if steps % args.test_interval == 0:  # 每隔一定的轮数，打印一次验证信息
                dev_acc = eval(dev_iter, model, args)  # 计算验证集准确率
                if dev_acc > best_acc:  # 如果验证集准确率大于最佳验证集准确率，则保存模型
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print(
                            'Saving best model, acc: {:.4f}%\n'.format(best_acc))  # 打印最佳验证集准确率
                        save(model, args.save_dir, 'best', steps)  # 保存模型
                else:
                    if steps - last_step >= args.early_stopping:  # 如果验证集准确率不再提升，则提前终止训练
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(
                            args.early_stopping, best_acc))  # 打印提前终止信息
                        raise KeyboardInterrupt


def eval(data_iter, model, args):
    model.eval()  # 模型设置为评估模式
    corrects, avg_loss = 0, 0  # 记录预测正确的个数和平均损失
    for batch in data_iter:  # 验证集迭代器，每次小批量验证
        feature, target = batch.text, batch.label  # 获取特征feature和目标target
        with torch.no_grad():
            feature.t_(), target.sub_(1)  # 将target转换为one-hot编码
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)  # 计算模型的输出
        loss = F.cross_entropy(logits, target)  # 计算损失
        avg_loss += loss.item()  # 记录平均损失
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()  # 计算准确率
    size = len(data_iter.dataset)  # 记录数据集大小
    avg_loss /= size  # 计算平均损失
    accuracy = 100.0 * corrects / size  # 计算准确率
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))  # 打印验证信息
    return accuracy


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)  # 保存模型的路径
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)  # 保存模型的名称
    torch.save(model.state_dict(), save_path)  # 保存模型
