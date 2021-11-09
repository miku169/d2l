import torch
# from my_data_set import MyDataset
# from torchvision.transforms import Compose, ToTensor
# from torch.utils.data import DataLoader
from torch import nn
# from d2l import torch as d2l


class Accumulator:  # @save
    """在`n`个变量上累加。"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # BERT微调所需的（之后将介绍）
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def print_net(net, input_size):
    a = torch.randn(size=input_size, dtype=torch.float32)
    print(a.shape)
    for layer in net:
        a = layer(a)
        print(f'{layer.__class__.__name__}\t{a.shape}')


def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，范例数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])  # loss,acc,len
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        print(f'train loss:{train_l:.3f},train acc:{train_acc:.3f}')
        if test_iter is not None:
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            print(f'test acc:{test_acc:.3f}')


# if __name__ == '__main__':
#     my_net = nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(784, 10)
#     )
#     # batch_size = 64
#     # trans = Compose([ToTensor()])
#     # dataset = MyDataset(root='../data/classify-leaves/', csv_path='train.csv', transform=trans)
#     # train_iter = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
#     device_gpu = torch.device('cuda:0')
#     # print_net(my_net, input_size=(1, 1, 224, 224))
#     '''
#     test
#     '''
#     batch_size = 256
#     num_epochs = 10
#     lr = 0.1
#     train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#     train(my_net, train_iter, test_iter, num_epochs, lr, device=device_gpu)
