import matplotlib.pyplot as plt


def draw(data, c, path):
    # 可视化散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 设置标签
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')

    # 绘制散点图
    # ax.scatter(data[:, 0], data[:, 1])
    ax.scatter(data[:, 0], data[:, 1], c=c)
    plt.savefig(path)
    plt.show()
    plt.close(fig)

def draw3d(data, c, path):
    # 可视化散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置标签
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_zlabel('Feature 2')

    # 绘制散点图
    # ax.scatter(data[:, 0], data[:, 1], data[:,2])
    ax.scatter(data[:, 0], data[:, 1], data[:,2], c=c)

    plt.savefig(path)
    plt.show()
    plt.close(fig)

def save_label(p, labels):
    with open(p, 'w') as f:
        i = 0
        for v in labels:
            ss = str(v)+' '
            i += 1
            if i >= 40:
                i = 0
                ss += "\n"
            f.write(ss)
