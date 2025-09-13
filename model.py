import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GATConv, GAT
from torch_geometric.utils import to_networkx
from DataLoader import *
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import random
device = torch.device("cpu")
import sys



class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"

        # Q 来自输入 A，K/V 来自输入 B
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask=None):
        """
        x_a: [batch, seq_len, embed_dim] 作为 Query
        x_b: [batch, seq_len, embed_dim] 作为 Key/Value
        """
        B, L, D = x_a.size()

        # Linear projection
        Q = self.q_proj(x_a)  # [B, L, D]
        K = self.k_proj(x_b)  # [B, L, D]
        V = self.v_proj(x_b)  # [B, L, D]

        # 分多头
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, Hd]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L, L]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, L, L]
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # [B, H, L, Hd]

        # 拼回去
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        out = self.out_proj(out)  # [B, L, D]

        return out, attn_weights



class SimpleCrossAttention(nn.Module):
    def __init__(self, d_model):
        super(SimpleCrossAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** 0.5
        self.linear = nn.Linear(d_model*2, d_model)

    def forward(self, x, y):
        """
        x: Tensor (seq_len, d_model) -> query
        y: Tensor (seq_len, d_model) -> key, value
        """
        Q = self.query(x)      # (seq_len, d_model)
        K = self.key(y)        # (seq_len, d_model)
        V = self.value(y)      # (seq_len, d_model)

        # 计算注意力权重 (seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.T) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 加权求和 (seq_len, d_model)
        out = torch.matmul(attn_weights, V)

        # 拼接原始 y
        out = torch.cat([out, y], dim=-1)  # (seq_len, d_model*2)
        out = self.linear(out)
        return out, attn_weights





class DiffGNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1):
        super(DiffGNNLayer, self).__init__()
        self.cross_attn1 = SimpleCrossAttention(d_model = output_dim * 2)
        self.cross_attn2 = SimpleCrossAttention(d_model = output_dim * 2)
        self.cross_attn3 = SimpleCrossAttention(d_model = output_dim * 2)
        # self.linear4 = nn.Linear((input_dim-9) * num_heads, output_dim)
        # self.SelfAttention = SelfAttention((input_dim-9), num_heads)
        self.conv1 = GATConv(33, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, output_dim * 2, heads=num_heads)

        self.conv3 = GATConv(64, hidden_dim, heads=num_heads)
        self.conv4 = GATConv(hidden_dim * num_heads, output_dim * 2, heads=num_heads)

        self.linear1 = nn.Linear(output_dim * 8, output_dim * 2)
        # self.linear1 = nn.Linear(input_dim + output_dim, output_dim * 2)

        self.linear2 = nn.Linear(output_dim * 4, output_dim)
        self.linear3 = nn.Linear(output_dim, 1)


        self.linear4 = nn.Linear(896, output_dim * 2)   # reaction层级的输出转换
        self.linear5 = nn.Linear(1, output_dim * 2)   # 量子层级的输出转换

    #
    def quantum_level(self, m, a, layer):
        # print("mmmmm",m.shape)
        h = m
        for n in range(layer):
            # print('a', n, a)
            adj = self.transform_tensor(a ** (n + 1)) / (10 ** n)
            # print('adj', n, adj)
            h_diff = h.unsqueeze(1) - h.unsqueeze(0)  # (num_nodes, num_nodes, in_feats)
            h_diff = h_diff * adj.unsqueeze(2)
            agg_diff = h_diff.sum(dim=1)
            h = agg_diff
            # print("hhhh",h.shape)

            # print('h', n, h)
        h = self.linear5(h)
        return h

    def atom_level(self, m, edge_index):
        h = self.conv1(m, edge_index)
        h = self.conv2(h, edge_index)
        return h

    def motif_level(self, m, edge_index):
        h = self.conv3(m, edge_index)
        h = self.conv4(h, edge_index)
        return h
    def reaction_level(self, m, edge_index):
        h = m
        return h

    # def reaction_level(self, ):

    def transform_matrix(sfle, matrix):  #
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] /= 10
        return matrix

    def transform_tensor(sfle, matrix):
        """
        将二维张量中的对角线变成0，并且其他非零值变为1
        参数:
        tensor (torch.Tensor): 输入的二维张量
        返回:
        torch.Tensor: 处理后的张量
        """
        if matrix.dim() != 2:
            raise ValueError("输入张量必须是二维的")
        # 生成单位矩阵并将对角线置为0
        mask = torch.ones_like(matrix) - torch.eye(matrix.size(0), dtype=torch.float32)
        # 将非零元素变为1
        binary_tensor = (matrix != 0).float()
        # 将对角线置为0，并保持其他非零元素为1
        result_tensor = binary_tensor * mask
        return result_tensor

    def sample_and_label_train(self, out_, y_list):
        l1 = y_list[0].tolist()
        l2 = y_list[1].tolist()
        sample_list = []
        condition = out_[int([element for element in y_list[1].tolist() if element not in y_list[0].tolist()][0])]
        p = []
        negative_index = []
        count = 0
        n = 0
        for ind2 in range(len(out_)):
            if ind2 in l2:
                # print(ind2)
                continue
            sample = torch.cat((out_[int(ind2)], condition), dim=0)
            sample_list.append(sample.unsqueeze(0))

            if ind2 in l1 and ind2 not in l2:
                p.append(sample.unsqueeze(0))
            else:
                negative_index.append(count)
            count += 1
        n1 = sample_list[random.choice(negative_index)]
        n2 = sample_list[random.choice(negative_index)]
        # print('ssssssssssssssssssssssss')
        return torch.cat([p[0], n1], dim=0), torch.tensor([[1], [0]], dtype=torch.float32)


    def sample_and_label_test(self, out_, y_list):
        l1 = y_list[0].tolist()
        l2 = y_list[1].tolist()
        label_list = []
        sample_list = []
        condition = out_[int([element for element in y_list[1].tolist() if element not in y_list[0].tolist()][0])]
        # p = 0

        for ind2 in range(len(out_)):
            if ind2 in l2:
                # print(ind2)
                continue
            sample = torch.cat((out_[int(ind2)], condition), dim=0)
            sample_list.append(sample.unsqueeze(0))

            if ind2 in l1 and ind2 not in l2:
                label_list.append(1)
            else:
                label_list.append(0)
        return torch.cat(sample_list, dim=0), torch.tensor(label_list, dtype=torch.float32)

    def forward(self, data, symbol):
        x = data.x
        y = data.y
        edge_index = data.edge_index
        tensor1_motif, tensor2_atom, tensor3_en,tensor4_reaction = torch.split(x, [64,33,1,896], dim=1)

        adj = to_dense_adj(data.edge_index)[0]

        h_motif = self.motif_level(tensor1_motif,edge_index)
        h_atom = self.atom_level(tensor2_atom, edge_index)
        h_en = self.quantum_level(tensor3_en, adj, 2)
        h_reaction = self.linear4(self.reaction_level(tensor4_reaction,edge_index))

        # 交叉注意力模块
        # print(tyh_motif))
        # print(type(h_atom))
        # print(type(h_en))
        # print("h_reaction",h_reaction)

        h_atom_cross, _ = self.cross_attn1(h_en, h_atom)

        h_motif_cross, _ = self.cross_attn2(h_atom_cross, h_motif)

        h_reaction_cross, _ = self.cross_attn3(h_motif_cross, h_reaction)
        # print("h_reaction_cross", h_reaction_cross)

        # concatenated_tensor = torch.cat((h_en, h_atom_cross, h_atom, h_motif_cross,h_motif, h_reaction_cross,h_reaction), dim=1)
        concatenated_tensor = torch.cat((h_en, h_atom_cross, h_motif_cross, h_reaction_cross), dim=1)

        h = self.linear1(concatenated_tensor)
        # print(h.shape)
        # print(h.shape)
        h = F.sigmoid(h)


        if symbol == "train":
            h, la = self.sample_and_label_train(h, y)
            # h, la = self.sample_and_label_test(h, y)
            # print(h)
        elif symbol == "test":
            h, la = self.sample_and_label_test(h, y)
        else:
            la = [None]

        h = self.linear2(h)
        h = self.linear3(h)
        h = F.sigmoid(h)
        return h, la


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, input, label):
        numerator = torch.tensor(0.0, dtype=torch.float)
        positive_list = [(input[int(label[0][0])], input[int(label[0][1])])]
        for u, v in positive_list:
            value = torch.norm(u-v, p=2)
            numerator += value
            # print('1', value.item())
            # print('p', u)
            # print('r', v)
            # if value == 0:
                # print('1', value.item())
                # print('p', u)
                # print('r', v)
            # print(numerator.item())
        if int(label[1][1]) == -1:
            denominator = torch.tensor(1.0, dtype=torch.float)
        else:
            negative_list = [(input[int(label[1][0])], input[int(label[1][1])])]
            denominator = torch.tensor(0.0, dtype=torch.float)
            for u, v in negative_list:
                value = torch.norm(u - v, p=2)
                denominator += value
                # print('2', value.item())
                # print('2', value.item())
                # print(denominator.item())
            # print('p', numerator)
            # print('r', denominator)
        # loss = numerator/denominator
        loss = numerator
        return loss


def remove_common_elements(list1, list2):
    # 找到两个列表中都出现的元素
    common_elements = set(list1) & set(list2)
    # 从两个列表中移除这些共同元素
    list1 = [item for item in list1 if item not in common_elements]
    list2 = [item for item in list2 if item not in common_elements]
    # print(list1)
    # print(list2)
    return int(list1[0]), int(list2[0])


def Judge_Predict_result(input_, list2):
    one_dim_tensor = input_.view(-1)  # or tensor.flatten()
    list1 = one_dim_tensor.tolist()
    # print('list1:',list1)
    # print('list2:',list2)
    # print(len(list1))
    # print(len(list2))
    max_value = max(list1)
    # print(max_value)
    max_indices = [index for index, value in enumerate(list1) if value == max_value]
    re = [list2[index] for index in max_indices]
    return re[0]


FilePath = 'Data/data-test.txt'
train_data, test_data = DataLoad(FilePath)

print('train_data:', len(train_data))
print('test_data:', len(test_data))
# 定义模型
input_dim = next(iter(train_data)).x.shape[1]
hidden_dim = 8
output_dim = 16
num_heads = 1  # 选择多头注意力机制的头数
num_epochs = 200  # 设置训练的轮次
# model = GATModel(input_dim, hidden_dim, output_dim, num_heads).to(device)

model = DiffGNNLayer(input_dim, hidden_dim, output_dim, num_heads)
# criterion = MyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
txt = open('model_OUT.txt', 'w', encoding='UTF-8')


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    # Train
    trainsample = []
    labellist = []
    for data in train_data:
        # print(data.x)
        data = data.to(device)
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        out, label = model(data, 'train')

        trainsample.append(out)
        labellist.append(label)
        # loss = criterion(out[int(label[0][0])], out[int(label[0][1])])
        # print(out)
        # print(label)
    sample = torch.cat(trainsample , dim=0)
    labels = torch.cat(labellist, dim=0)

    # print('labels', sum(labels))
    loss = criterion(sample, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    # Test
    rightcount = 0
    count = 1
    for test in test_data:
        out, label = model(test, 'test')
        count += 1
        # print(out)
        # print(sum(label.tolist()))
        result = Judge_Predict_result(out, label)
        rightcount += result
    # 打印当前 epoch 的平均损失
    # if total_loss == 0 or str(total_loss / len(train_data)) == 'nan':
    #     print(total_loss)
    #     break
    # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_data)}, Accuracy: {rightcount / len(test_data)}")
    txt.write(str(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_data)}, Accuracy: {rightcount / len(test_data)}") + '\n')
print("训练完成！")
