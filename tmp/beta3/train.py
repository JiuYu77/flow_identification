# -*- coding:UTF-8 -*-
import torch
from torch import optim
import numpy as np
import sys
sys.path.append('.')

from tmp.beta3.beta import BetaVAE1D, loss_function
from jyu.utils import data_loader, FlowDataset, tu

device = tu.get_device()
print(device)

# 定义模型和优化器
model = BetaVAE1D(input_length=4096, latent_dim=20, beta=4.0)  # 可以调整beta值
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
model.to(device)

datasetPath = "/home/jyu/apro/flow/dataset/flow/v4/Pressure/v4/train"
train_loader = data_loader(FlowDataset, datasetPath, 4096, 2048, "normalization_MinMax", supervised=False, batchSize=64, shuffle=True, numWorkers=4)
path = "tmp/beta3/"

# 训练过程
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, _, index) in enumerate(train_loader):
        # print(data.shape)
        print("\033[Kbatch_idx: ", batch_idx, end='\r')
        data = data.to(device)

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, beta=model.beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}')
    torch.save(model.state_dict(), path+"min_loss_params.pt")
torch.save(model.state_dict(), path+"last_params.pt")


# 提取潜在表示
def extract_latent_space(model, dataloader):
    model.eval()
    latent_vectors = []
    idx = []
    with torch.no_grad():
        for data,_,index in dataloader:
            data = data.to(device)
            mu, logvar = model.encode(data)
            z = model.reparameterize(mu, logvar)  # 或直接使用 mu
            latent_vectors.append(z.cpu().numpy())
            for v in index:
                idx.append(int(v))
    return np.concatenate(latent_vectors, axis=0), idx

latent_space, idx = extract_latent_space(model, train_loader)
print(f"Latent space shape: {latent_space.shape}")  # (num_samples, latent_dim)

from sklearn.cluster import KMeans

# 设置簇的数量 k
num_clusters = 7
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
# kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(latent_space)

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

# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(latent_space_2d[:, 0], latent_space_2d[:, 1], c=train_loader.dataset.allLabel, cmap='viridis', s=10)
# plt.colorbar(scatter)
# plt.title('Clustering Results in Latent Space')
# plt.xlabel('t-SNE Dim 1')
# plt.ylabel('t-SNE Dim 2')
# plt.savefig(path+'true')
