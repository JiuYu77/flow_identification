import torch
import numpy as np
import sys
sys.path.append('.')

from tmp.beta3.beta4 import BetaVAE1D
from jyu.torch_utils import tu
from jyu.dataset.flowDataset import FlowDataset
from jyu.dataloader.dataLoader_torch import data_loader
from jyu.tml import *


path = "tmp/beta3/"
datasetPath = "/home/jyu/apro/my_git/dataset/flow/v4/Pressure/v4/train"
device = tu.get_device()

model = BetaVAE1D(1, latent_dim=3)
model.to(device)

f = path + "min_loss_params.pt"
# f = path + "last_params.pt"
stateDict = torch.load(f, device)
model.load_state_dict(stateDict)

train_loader = data_loader(FlowDataset, datasetPath, 8192, 2048, "normalization_MinMax", supervised=False, batchSize=64, shuffle=False, numWorkers=4)
# indices = train_loader.dataset.do_shuffle()

# 提取潜在表示
def extract_latent_space(model, dataloader):
    model.eval()
    latent_vectors = []
    idx = []
    with torch.no_grad():
        for data, _, index in dataloader:
            data = data.to(device)
            mu, logvar = model.encode(data)
            z = model.reparameterize(mu, logvar)  # 或直接使用 mu
            latent_vectors.append(z.cpu().numpy())
            for v in index:
                idx.append(int(v))
    return np.concatenate(latent_vectors, axis=0), idx

latent_space, idx = extract_latent_space(model, train_loader)
print(f"Latent space shape: {latent_space.shape}")  # (num_samples, latent_dim)


# 设置簇的数量 k
num_clusters = 7
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
# kmeans = KMeans(n_clusters=num_clusters)
# cluster_labels = kmeans.fit_predict(latent_space)

# cluster_labels = do_k_means(latent_space, num_clusters)
cluster_labels = do_k_means(latent_space, num_clusters, 'auto', random_state=None)

# cluster_labels = do_MiniBatchKMeans(latent_space, num_clusters, 'auto', 128, 1000)
# idx = indices

print(f"Cluster labels: {cluster_labels}")

pre_labels = [0 for _ in range(0, len(idx))]
for i,v in enumerate(idx):
    pre_labels[v] = cluster_labels[i]

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
save_label(path + "pre.txt", cluster_labels)
save_label(path + "pre_order.txt", pre_labels)
save_label(path + "true.txt", train_loader.dataset.allLabel)



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 使用 t-SNE 降维到 2D
print("降维...")
tsne = TSNE(n_components=2, random_state=42)
latent_space_2d = tsne.fit_transform(latent_space)

# 可视化聚类结果
def draw(x, c, p, title=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x[:, 0], x[:, 1], c=c, cmap='viridis', s=10)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.savefig(p)

print("draw...")
draw(latent_space_2d, cluster_labels, path+'pre', 'Clustering Results in Latent Space')
draw(latent_space_2d, pre_labels, path+'pre_order', 'Ordered Clustering Results in Latent Space')
draw(latent_space_2d, train_loader.dataset.allLabel, path+'true', 'True Label')

