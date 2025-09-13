import networkx as nx
from collections import defaultdict
from GetMoleculeFeatures import *
from get_motif import *
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import NormalizeFeatures
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd


# Step 1: 读取Excel数据
def load_graph_from_excel(file_path: str):
    node_df = pd.read_excel(file_path, sheet_name=0)
    edge_df = pd.read_excel(file_path, sheet_name=1)

    node_ids = node_df.iloc[:, 0].values
    node_structures = node_df.iloc[:, 1].values

    # 如果节点特征是字符串或结构体，进行简单编码
    if node_structures.dtype == object:
        unique_features = list(set(node_structures))
        feature_to_idx = {feat: idx for idx, feat in enumerate(unique_features)}
        node_features = torch.tensor([feature_to_idx[feat] for feat in node_structures], dtype=torch.long)
        node_features = F.one_hot(node_features).to(torch.float)
        # print(len(node_features))
        # print(node_features)
    else:
        node_features = torch.tensor(node_structures, dtype=torch.float).unsqueeze(1)

    edge_index = torch.tensor(edge_df.iloc[:, :2].values.T, dtype=torch.long)
    edge_weight = torch.tensor(edge_df.iloc[:, 2].values, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
    # print(node_structures)
    return data, node_structures


# Step 2: 定义两层GCN模型
class TwoLayerGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(TwoLayerGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


# Step 3: 主执行逻辑
def get_motif_GCN_feature_dict (file_path: str):  # 获取motif列表被GCN提取之后的整体特征
    data, node_structures = load_graph_from_excel(file_path)

    model = TwoLayerGCN(
        in_channels=data.num_node_features,
        hidden_channels=16,
        out_channels=8
    )

    model.eval()
    with torch.no_grad():
        node_embeddings = model(data)
    # print(node_embeddings)
    node_dict = {structure: embedding for structure, embedding in zip(node_structures, node_embeddings)}
    return node_dict


def pad_list_to_equal_length(list1, list2):
    len1 = len(list1)
    len2 = len(list2)
    max_length = max(len1, len2)
    if len1 < max_length:
        list1 += [-1] * (max_length - len1)
    if len2 < max_length:
        list2 += [-1] * (max_length - len2)
    return [list1, list2]


def graph_to_edge_index(graph, Nodes_name_index_dict):
    # 获取图的所有边
    edges = list(graph.edges())
    # print(edges)
    # 初始化 edge_index 列表
    edge_index = []
    # 遍历所有边
    for i, j in edges:
        # 将每条边添加到 edge_index 列表中
        u = Nodes_name_index_dict[i]
        v = Nodes_name_index_dict[j]
        edge_index.append((u, v))
        edge_index.append((v, u))
    # 将 edge_index 列表转换为 PyTorch 张量，并转置
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    return edge_index


def DataProcess(Molecule_Smiles, num, motif_feature_dict, reaction_dict,r):
    print(Molecule_Smiles)
    mol = Chem.MolFromSmiles(Molecule_Smiles)

    atom_to_motif = generate_mapping_table(mol, list(motif_feature_dict.keys())[0:-1])

    node_features = defaultdict(lambda: torch.tensor([]))
    topology_graph = nx.Graph()
    conut = 0
    Label_protect = []
    Label_reacte = []
    Label_pair_P_R = []
    node_name = []
    feature3_reaction = reaction_dict[f"{r.rstrip()}|{Molecule_Smiles.rstrip()}|{0}"]
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()  # 获取原子的索引
        # if atom_to_motif[atom_index] == "None":
        #     feature1_motif =
        # else:
        feature1_motif = motif_feature_dict[atom_to_motif[atom_index]]
        atom_id = str(num) + f'_atom_{atom_index}'  # 为原子创建唯一 ID
        node_name.append(atom_id)
        topology_graph.add_node(atom_id)
        feature2_atom_en = torch.tensor(GetAtomFeature(atom, mol))

        # print(feature2_atom_en.shape)
        # print(feature1_motif)
        # print(feature2)
        conut += 1
        if 100 <= int(atom.GetAtomMapNum()) < 200:
            Label_protect.append(atom_id)
            if str(atom.GetAtomMapNum())[-1] == '5':
                Label_pair_P_R.append(atom_id)
        elif 200 <= int(atom.GetAtomMapNum()) < 1000:
            Label_reacte.append(atom_id)
            if str(atom.GetAtomMapNum())[-1] == '5':
                Label_pair_P_R.append(atom_id)
        node_features[atom_id] = torch.cat((feature1_motif, feature2_atom_en, feature3_reaction), dim=0)
        key = f"{r.rstrip()}|{Molecule_Smiles.rstrip()}|{atom_index+1}"
        if key not in reaction_dict.keys():
            continue
        feature3_reaction = reaction_dict[f"{r.rstrip()}|{Molecule_Smiles.rstrip()}|{atom_index+1}"]


    # 遍历分子的化学键
    for bond in mol.GetBonds():
        # bond_index = bond.GetIdx()  # 获取化学键的索引
        # bond_id = str(num) + f'_bond_{bond_index}'  # 为化学键创建唯一 ID
        # node_name.append(bond_id)
        # topology_graph.add_node(bond_id)
        # bondfeature = GetBondFeature(bond)
        # node_features[bond_id] = bondfeature
        # 获取化学键连接的原子
        begin_atom_id = str(num) + f'_atom_{bond.GetBeginAtomIdx()}'
        end_atom_id = str(num) + f'_atom_{bond.GetEndAtomIdx()}'
        # 在二部图中添加边，连接化学键和原子
        # topology_graph.add_edge(bond_id, begin_atom_id)
        # topology_graph.add_edge(bond_id, end_atom_id)
        # 只连接原子，构建没有键的拓扑图。和上面的构建二部图中连接化学键和原子的代码二选一，同时不能添加边的节点、特征等等。
        topology_graph.add_edge(end_atom_id, begin_atom_id)
        # conut += 1
        # if bond.GetBeginAtom().GetAtomMapNum() > 1000 and bond.GetEndAtom().GetAtomMapNum() > 1000:
        #     Label_protect.append(bond_id)
        #     if str(bond.GetBeginAtom().GetAtomMapNum())[-1] == '5' and str(bond.GetEndAtom().GetAtomMapNum())[-1] == '5':
        #         Label_pair_P_R.append(bond_id)

    # print(node_set)
    # print(f"图的节点数量：{topology_graph.number_of_nodes()}")
    # print(f"图的边数量：{topology_graph.number_of_edges()}")
    return node_name, node_features, topology_graph, Label_protect, Label_reacte, Label_pair_P_R

# def GetLabelIndex(s):

def Sample_creation(S_list,motif_feature_dict,reaction_dict,r):
    merged_graph = nx.Graph()
    Feature = defaultdict(dict)
    Label_Protect = []
    Label_Reacte = []
    Label_pair_Protect_Reacte = []
    name_list = []  # 得到所有节点的名称及先后顺序
    for i in range(len(S_list)):
        name, feature, graph, Label_p, Label_r, Label_p_r = DataProcess(S_list[i], i,motif_feature_dict,reaction_dict,r)
        merged_graph.update(graph)
        Feature.update(feature)
        Label_Protect += Label_p
        Label_Reacte += Label_r
        Label_pair_Protect_Reacte += Label_p_r
        name_list += name

    Nodes_name_index_dict = {}
    count = 0
    for t in name_list:
        Nodes_name_index_dict[t] = float(count)
        count += 1
    Label_Protect_Index = [Nodes_name_index_dict[id_] for id_ in Label_Protect]
    Label_Reacte_Index = [Nodes_name_index_dict[id_] for id_ in Label_Reacte]
    Label_pair_Protect_Reacte_Index = [Nodes_name_index_dict[id_] for id_ in Label_pair_Protect_Reacte]
    edge_index_ = graph_to_edge_index(merged_graph, Nodes_name_index_dict)
    Feature_index = {}
    for u in name_list:
        Feature_index[Nodes_name_index_dict[u]] = Feature[u]
    return Feature_index, edge_index_, pad_list_to_equal_length(Label_pair_Protect_Reacte_Index, Label_Reacte_Index)



def DataLoad(FilePath):

    # motif 特征提取
    # motif_feature_dict = get_motif_GCN_feature_dict("Data/output-50K-Data.xlsx")   #
    df = pd.read_excel("Data/motif_embeddings.xlsx")
    # 构建字典：{label: tensor([...])}
    motif_feature_dict = {
        row['label']: torch.tensor([float(x) for x in row['embedding'].split(",")], dtype=torch.float32)
        for _, row in df.iterrows()
    }

    # reaction 特征提取
    df = pd.read_excel("Data/reaction_atom_features_reactants.xlsx")

    # 构建字典
    reaction_dict = {}

    for _, row in df.iterrows():
        reaction = row["reaction"]
        reactant = row["reactant_smiles"]
        atom_idx = int(row["atom_idx"])

        # 构造 key
        key = f"{reaction}|{reactant}|{atom_idx}"

        # 取出 embedding 向量 (feat_0, feat_1, ...)
        feat_cols = [col for col in df.columns if col.startswith("feat_")]
        # 显式转换为 float
        embedding = torch.tensor(row[feat_cols].astype(float).values, dtype=torch.float32)

        reaction_dict[key] = embedding

    #
    Reaction_list = []
    with open(FilePath, 'r', encoding='UTF-8') as file_object:
        for line in file_object:
            Reaction_list.append(line)
    print('Number_Data:', len(Reaction_list))
    # print(Reaction_list[0])
    Datalist = []
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(motif_feature_dict)
    for i in range(len(Reaction_list)):
        r = Reaction_list[i]
        Reactants, Products = r.split('>>')
        Reactants_list = Reactants.split('.')
        Feature_index, edge_index_, Label_list = Sample_creation(Reactants_list,motif_feature_dict,reaction_dict,r)
        # print(edge_index_)
        Fe = []
        for q in range(len(Feature_index)):
            Fe.append(Feature_index[q])
            # print(f"节点 {q} 的初始特征：{Feature_index[q]}")
        # print(Fe)
        F_tensor = F.normalize(torch.stack(Fe, dim=0), p=2, dim=0)
        data = Data(x=F_tensor, edge_index=edge_index_, y=torch.tensor(Label_list))

        # data = Data(x=F_tensor, edge_index=edge_index_, y=torch.tensor(Label_list))

        # data = Data(x=torch.tensor(Fe, dtype=torch.float), edge_index=edge_index_, y=torch.tensor(Label_list))
        Datalist.append(data)
        # print(F_tensor)

    train_indices, test_indices = train_test_split(range(len(Datalist)), test_size=0.25, random_state=42)
    train_data_list = [Datalist[j] for j in train_indices]
    test_data_list = [Datalist[j] for j in test_indices]
    train_dataloader = DataLoader(train_data_list, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data_list, batch_size=1, shuffle=False)

    return train_dataloader, test_dataloader

# FilePath = 'Data/Data.txt'
# train_data, test_data = DataLoad(FilePath)
# first_batch = next(iter(train_data))
# print(first_batch.x)


