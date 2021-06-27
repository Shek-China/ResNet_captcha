# Defined in file: ./chapter_preface/index.md
import time
from IPython import display
from matplotlib import pyplot as plt
import os
from PIL import Image
import tqdm

# Defined in file: ./chapter_preface/index.md
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

charNumber = 5
ImageHeight = 130
ImageWidth = 260


# Defined in file: ./chapter_preliminaries/calculus.md
def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def StrtoLabel(Str):
    # convert to int, [0, 61]
    label = []
    for i in range(0, charNumber):
        if Str[i] >= '0' and Str[i] <= '9':
            label.append(ord(Str[i]) - ord('0'))
        elif Str[i] >= 'a' and Str[i] <= 'z':
            label.append(ord(Str[i]) - ord('a') + 10)
        else:
            label.append(ord(Str[i]) - ord('A') + 36)
    return label


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
class Animator:
    """For plotting data in animation."""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[
                                                0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class Captcha(data.Dataset):
    def __init__(self, root, train=True):
        self.imgsPath = [os.path.join(root, img) for img in os.listdir(root)]
        self.transform = transforms.Compose([
            transforms.Resize((ImageHeight, ImageWidth)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # transforms.Lambda(lambda x: x.repeat(1, 1, 1))
        ])

    def __getitem__(self, index):
        imgPath = self.imgsPath[index]
        # print(imgPath)
        label = imgPath.split("/")[-1].split(".")[0]
        # print(label)
        labelTensor = torch.Tensor(StrtoLabel(label))
        data = Image.open(imgPath).convert('RGB')
        # print(data.size), [64, 24]
        data = self.transform(data)
        # print(data.shape), [3, 32, 32]
        return data, labelTensor

    def __len__(self):
        return len(self.imgsPath)


class Timer:
    """Record multiple running times."""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        net.train()
        #for i, (X, y) in tqdm.tqdm(enumerate(train_iter)):
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_label1, y_label2, y_label3 = y[:, 0], y[:, 1], y[:, 2]
            # y_hat = net(X)
            y = y.long()
            y1, y2, y3 = net(X)
            l1, l2, l3 = loss(y1, y_label1) + loss(y2, y_label2) + loss(y3, y_label3)
            # l = loss(y_hat, y)
            l = l1 + l2 + l3

            l.backward()
            optimizer.step()
            with torch.no_grad():
                # metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
                metric.add(loss.item())
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


# def load_data_fashion_mnist(batch_size, resize=None):
#     """Download the Fashion-MNIST dataset and then load it into memory."""
#     trans = [transforms.ToTensor()]
#     if resize:
#         trans.insert(0, transforms.Resize(resize))
#     trans = transforms.Compose(trans)
#     mnist_train = torchvision.datasets.FashionMNIST(root="../data",
#                                                     train=True,
#                                                     transform=trans,
#                                                     download=True)
#     mnist_test = torchvision.datasets.FashionMNIST(root="../data",
#                                                    train=False,
#                                                    transform=trans,
#                                                    download=True)
#     return (data.DataLoader(mnist_train, batch_size, shuffle=True,
#                             num_workers=get_dataloader_workers()),
#             data.DataLoader(mnist_test, batch_size, shuffle=False,
#                             num_workers=get_dataloader_workers()))


def load_data(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trainDataset = Captcha("./train/", train=True)  # load img path
    testDataset = Captcha("./test/", train=False)
    return (data.DataLoader(trainDataset, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(testDataset, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # Required for BERT Fine-tuning (to be covered later)
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]


def get_dataloader_workers():
    """Use 4 processes to read the data."""
    return 4


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
