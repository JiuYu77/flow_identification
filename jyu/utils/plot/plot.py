# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
# import matplotlib; matplotlib.use('TkAgg')
from matplotlib import font_manager
import os
import numpy as np
import itertools

COLOR = {
    'darkblue':'#0072BD',
    'darkorange':'#FF7900',
}

def set_matplotlib():
    """设置matplotlib"""
    # mpl.use('TkAgg')  # 在一个新窗口打开图形
    font_path = os.path.join(os.getcwd(), 'jyu', 'utils', 'SimHei.ttf')  # Your font path goes here
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = prop.get_name()
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def set_axes(axes:plt.Axes,
             xlabel=None, ylabel=None,
             xlim=None, ylim=None,
             xscale=None, yscale=None,
             title=None, legend=None, grid=True):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel) if xlabel else ...
    axes.set_ylabel(ylabel) if ylabel else ...
    axes.set_xlim(xlim) if xlim else ...
    axes.set_ylim(ylim) if ylim else ...
    axes.set_xscale(xscale) if xscale else ...
    axes.set_yscale(yscale) if yscale else ...
    axes.set_title(title) if title else ...
    if legend:
         axes.legend(legend)
    if grid:
        axes.grid()

def set_plt(xlabel=None, ylabel=None, xlim=None, ylim=None,
                 xscale=None, yscale=None, title=None, legend=None, grid=True):
    """设置matplotlib的轴"""
    plt.xlabel(xlabel) if xlabel else ...
    plt.ylabel(ylabel) if ylabel else ...
    plt.xlim(xlim) if xlim else ...
    plt.ylim(ylim) if ylim else ...
    plt.xscale(xscale) if xscale else ...
    plt.yscale(yscale) if yscale else ...
    plt.title(title) if title else ...
    if legend:
        plt.legend(legend)
    if grid:
        plt.grid()


class Plot:
    def __init__(self, figsize=None) -> None:
        width, height = 3.5, 2.5
        self.figsize = (width, height)
        if figsize:
            self.figsize = figsize

    def get_figure(self, figsize=None):
        if figsize is None:
            figsize = self.figsize
        return plt.figure(figsize=figsize)

    def init_axes(self, ax:plt.Axes):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    
    def get_axes(self, fig=None, nRows=1, nCols=1, index=1):
        if fig is None:
            fig = self.get_figure()
        ax = fig.add_subplot(nRows, nCols, index)
        self.init_axes(ax)
        return ax
 
    def getRange(self, start, end, step):
        ranges = []
        i = start
        while i <= end:
            ranges.append(i)
            i += step
        return ranges

    def draw(self):
        pass

    def draw_save(self, *args, **kwargs):
        """画图，保存"""
        self.draw()
        save(*args, **kwargs)
        close()

    def draw_show(self, *args, **kwargs):
        """画图，显示"""
        self.draw()
        show(*args, **kwargs)


def save(*args, **kwargs):
    """save image"""
    plt.savefig(*args, **kwargs)

def show(*args, **kwargs):
    """show image"""
    plt.show(*args, **kwargs)

def close(fig=None):
    """Close a figure window."""
    plt.close(fig)

class Line(Plot):
    """绘制折线图"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-', 'r:'),
                 nrows=1, ncols=1, figsize=(3.5, 2.5)) -> None:
        if legend is None:
            legend = []
        self.fig:plt.Figure
        self.axes:plt.Axes
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
             self.axes:list[plt.Axes] = [self.axes]
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend=legend)
        self.X, self.Y, self.fmts = None, None, fmts

        self.xlabel, self.ylabel, self.legend, self.xlim, self.ylim,\
        self.xscale, self.yscale, self.fmts, self.nrows, self.ncols, self.figsize = \
            xlabel, ylabel, legend, xlim, ylim, xscale, yscale, fmts, nrows, ncols, figsize

    def Init(self):
        xlabel, ylabel, legend, xlim, ylim, xscale, yscale, fmts, nrows, ncols, figsize = \
        self.xlabel, self.ylabel, self.legend, self.xlim, self.ylim,\
        self.xscale, self.yscale, self.fmts, self.nrows, self.ncols, self.figsize
        if legend is None:
            legend = []
        self.fig:plt.Figure
        self.axes:plt.Axes
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
             self.axes:list[plt.Axes] = [self.axes]
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend=legend)

    def add(self, x, y):
        """
        向图表中添加多个数据点
        x中的每个元素，对应每条曲线各自的横坐标
        y中的每个元素，对应每条曲线各自的纵坐标
        y中元素的个数，表示曲线的个数
        """
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
 
    def draw(self):
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()


def plot_one_curve(X,Y,xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None, imgPath=None,
                 xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-', 'r:'), figsize=(3.5, 2.5)):
    """
    绘制曲线/折线
    X是这样的列表：[]
    Y是这样的列表：[]
    """
    plt.figure(figsize=figsize)
    if X is None:  # 绘制y坐标，x坐标范围使用列表xlim
        plt.plot(Y)
    else:
        plt.plot(X, Y)
    set_plt(xlabel, ylabel, xlim, ylim, xscale, yscale, legend=legend)
    plt.savefig(imgPath)

def plot_curve(X,Y,xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None, imgPath=None,
                 xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-', 'r:'), figsize=(3.5, 2.5)):
    """
    绘制曲线/折线
    X是这样的列表：[[],[],[],...]
    Y是这样的列表：[[],[],[],...]
    """
    plt.figure(figsize=figsize)
    if X is None:  # 绘制y坐标，x坐标范围使用列表0..N-1
        xlim = None
        for y in Y:
            plt.plot(y)
    else:
        x = X[0]
        for _ in range(len(Y) -1):
            X.append(x)
        for x, y,fmt in zip(X, Y, fmts):
            plt.plot(x, y, fmt)
    set_plt(xlabel, ylabel, xlim, ylim, xscale, yscale, legend=legend)
    plt.savefig(imgPath)


class ConfusionMatrix(Plot):
    """混淆矩阵"""
    def __init__(self, classes,  normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(6, 5), fontdict=dict(fontsize=12)) -> None:
        """
        classes:tuple or list
        """
        super().__init__(figsize=figsize)
        self.cm = [[0 for _ in range(len(classes))] for _ in range(len(classes))]
        self.classes = classes
        self.normalize = normalize
        self.title = title
        self.cmap = cmap
        self.get_figure(figsize=figsize)
        self.fontdict = fontdict
        self.fontdict2 = dict(fontsize=fontdict['fontsize']-2)

    def add(self, trueIdx, preIdx):
        self.cm[trueIdx][preIdx] += 1

    def draw(self):
        self.cm = np.array(self.cm)
        self.plot_confusion_matrix(self.cm, self.classes, self.fontdict, self.fontdict2, self.normalize, self.title, self.cmap)

    @staticmethod 
    def plot_confusion_matrix(cm, classes, fontdict, fontdict2, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        """绘制混淆矩阵"""
        if normalize:
            # normalizing operation
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # get line sum,and expand one dimension
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # cmap means the color
        plt.imshow(cm, interpolation='nearest', cmap=cmap)

        plt.title(title, fontdict=fontdict)
        # color bar on the right
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontdict=fontdict)
        plt.yticks(tick_marks, classes, fontdict=fontdict)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            # print the number in cmt[j,i]
            plt.text(j, i, format(cm[i, j] * 100, fmt) + '%', horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black", fontdict=fontdict)

        plt.xlabel('Predicted label', fontdict=fontdict2)
        plt.ylabel('True label', fontdict=fontdict2)
        plt.tight_layout()
        # plt.show()

class Scatter(Plot):
    """散点图"""
    def __init__(self, path, xlabel=None, ylabel=None, s=None, c=None, marker=None, cmap=None, alpha=.5, figsize=(18, 12)) -> None:
        super().__init__(figsize=figsize)
        self.path = path
        self.xlabel, self.ylabel = xlabel, ylabel
        self.s, self.c, self.marker, self.cmap, self.alpha = s, c, marker, cmap, alpha
        self.x, self.y = [], []
        self._load_data()

    def _load_data(self):
        from utils.plot.getdata import ScatterData
        sd = ScatterData(self.path)
        step = 5
        self.trainLoss = sd.get_avgloss()[::step]
        self.valLoss = sd.get_avgloss(train=False)[::step]
        self.trainAcc = sd.get_acc()[::step]
        self.valAcc = sd.get_acc(False)[::step]
        end = len(self.trainLoss)
        self.trainX = self.getRange(1, end, 1)
        self.valX = self.getRange(1, end, end / len(self.valLoss))
 
    def draw(self):
        plt.scatter(self.x, self.y, self.s, self.c, self.marker, self.cmap, self.alpha)

    def draw_save(self):
        trainLoss, valLoss = self.trainLoss, self.valLoss
        trainAcc, valAcc = self.trainAcc, self.valAcc
        trainX, valX = self.trainX, self.valX
        plt.rcParams.update({'font.size': 20})
        axes = self.get_axes()
        alpha = 1.
        axes.scatter(trainX, trainLoss, c=COLOR['darkblue'], alpha=alpha)
        axes.scatter(valX, valLoss, c=COLOR['darkorange'], alpha=alpha)
        axes.text(2000, 1, 'loss scatter', fontdict=dict(fontsize=30))
        set_plt(ylim=[0, 2.52], legend=['train loss', 'val loss'], grid=False)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        lossPath = os.path.join(self.path, 'loss_scatter.png') 
        plt.savefig(lossPath)
        # plt.show(block=False)

        self.get_figure()
        alpha = 1.
        plt.scatter(trainX, trainAcc, c=COLOR['darkblue'], alpha=alpha)
        plt.scatter(valX, valAcc, c=COLOR['darkorange'], alpha=alpha)
        plt.text(2000, 0.5, 'acc scatter', fontdict=dict(fontsize=30))
        set_plt(ylim=[0, 1.02], legend=['train acc', 'val acc'], grid=False)
        accPath = os.path.join(self.path, 'acc_scatter.png') 
        plt.savefig(accPath)
        # plt.show()



if __name__ == '__main__':
    def Line_test():
        a = Line('x','y', legend=['le'], xlim=[0, 1.0])
        a.add(0.1, 0.1)
        a.add(0.2, 0.2)
        a.add(0.3, 0.4)
        a.add(0.5, 0.8)
        a.draw()
        # save('a.png')
        # ***********
        b = Line('x','y', legend=['01', '02', '03'], xlim=[0, 1.0], figsize=(5, 4))
        b.add(0.1, [0.2, 0.3, 0.5, 1])
        b.add(0.2, [0.3, 0.6, 0.9])
        b.add(0.5, [0.5, 0.8, 0.6])
        b.add(0.8, [0.6, 0.8, 1.2, 2])
        b.draw()
        # b.draw_show()
        # ***********
        c = Line('x','y', legend=['01', '02','03'], xlim=[0, 1.0], figsize=(5, 4))
        c.add([0.1, 0.2, 0.1], [0.1, 0.2, 0.3])
        c.add([0.2, 0.3, 0.4], [0.1, 0.5, 0.4])
        c.add([0.3, 0.5, 0.6], [0.2, 0.3])
        c.add([0.5, 0.6, 0.9], [0.2, 0.3, 0.6])
        c.draw()
        # c.draw_show()
        show()

    def Confusion():
        cm = np.array([
            [5624,  13,   68,    99,    10 ],
            [14,   5901,   4,    61,     4 ],
            [151,   6,    4721,  60,   712 ],
            [295,   42,    22,  5382,  154 ],
            [20,    9,    444 ,  230, 4892 ],
        ])
        names = (
            'T-shirt/top',
            'Trouser',
            'Pullover',
            'Dress',
            'Coat',
        )
        c = ConfusionMatrix(names, True)
        c.cm = cm
        c.draw()
    
    def scatter():
        n = 1024
        X = np.random.normal(0,1,n)
        Y = np.random.normal(0,1,n)
        T = np.arctan2(Y,X)  # T中包含了数据点的颜色到当前colormap的映射值
        plt.scatter(X, Y, c=T, alpha=.5)
        plt.show()

    # Line_test()
    # Confusion()
    scatter()
