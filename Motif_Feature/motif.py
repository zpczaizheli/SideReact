import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

# === Load Data ===
edges = pd.read_excel('edges.xlsx')
nodes = pd.read_excel('updated_nodes_with_classification.xlsx')

# === Reindex Nodes ===
node_ids = nodes['id'].unique()
id2idx = {id_: idx for idx, id_ in enumerate(node_ids)}
print(id2idx)
num_nodes = len(node_ids)

# === Create Adjacency Matrix ===
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
for _, row in edges.iterrows():
    i, j = id2idx[row['Source']], id2idx[row['Target']]
    adj_matrix[i, j] = row['weight']
    adj_matrix[j, i] = row['weight']

adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)

adj_norm = adj_tensor / adj_tensor.max()  # normalize by max
print(adj_tensor)
# === One-Hot Encode Node IDs ===
ohe = OneHotEncoder(sparse_output=False)
x_encoded = ohe.fit_transform(nodes[['id']])
x_tensor = torch.tensor(x_encoded.astype(np.float32))

# === GCN Layer ===
class GCNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim, 256)
        self.linear2 = torch.nn.Linear(256, out_dim)

    def forward(self, x, adj):
        x = adj @ self.linear1(x)
        x = F.sigmoid(x)
        x = adj @ self.linear2(x)
        return x

# === Training Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_tensor = x_tensor.to(device)
adj_norm = adj_norm.to(device)

model = GCNLayer(in_dim=x_tensor.shape[1], out_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# === Training Loop ===
for epoch in range(1, 1001):
    model.train()
    optimizer.zero_grad()
    z = model(x_tensor, adj_norm)
    pre_adj = torch.matmul(z, z.T)
    loss = F.mse_loss(pre_adj, adj_norm)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.20f}")

# === Extract Embeddings ===
model.eval()
with torch.no_grad():
    final_embeddings = model(x_tensor, adj_norm)
    # print(final_embeddings)
    # print(final_embeddings.shape)
    # print(num_nodes)
    # print(nodes[['id']])

    pre_adj = torch.matmul(final_embeddings, final_embeddings.T)
    r2 = r2_score(adj_norm.cpu().numpy().flatten(), pre_adj.cpu().numpy().flatten())
    print(f"R2: {r2:.4f}")


# === t-SNE Visualization ===
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
z_tsne = tsne.fit_transform(final_embeddings.cpu().numpy())

plt.figure(figsize=(8, 6))
types = nodes['type'].values
unique_types = np.unique(types)
type_color_map = {1: 'forestgreen', 2: 'deepskyblue', 3: 'darkorange'}

for t in unique_types:
    mask = types == t
    color = type_color_map.get(t, 'gray')
    plt.scatter(z_tsne[mask, 0], z_tsne[mask, 1], label=str(t), alpha=0.7, s=20, color=color)

plt.title("t-SNE of Node Embeddings by Type")
plt.legend(title="Node Type")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("tsne_node_embeddings_by_type2.png", dpi=1000)
plt.show()

# === Save Embeddings to Excel ===
embedding_df = pd.DataFrame({
    "label": nodes['label'].values,
    "embedding": [",".join(map(str, vec)) for vec in final_embeddings.cpu().numpy()]
})

embedding_df.to_excel("motif_embeddings.xlsx", index=False)

