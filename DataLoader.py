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



def load_graph_from_excel(file_path: str):
    node_df = pd.read_excel(file_path, sheet_name=0)
    edge_df = pd.read_excel(file_path, sheet_name=1)

    node_ids = node_df.iloc[:, 0].values
    node_structures = node_df.iloc[:, 1].values

    
    if node_structures.dtype == object:
        unique_features = list(set(node_structures))
        feature_to_idx = {feat: idx for idx, feat in enumerate(unique_features)}
        node_features = torch.tensor([feature_to_idx[feat] for feat in node_structures], dtype=torch.long)
        node_features = F.one_hot(node_features).to(torch.float)

    else:
        node_features = torch.tensor(node_structures, dtype=torch.float).unsqueeze(1)

    edge_index = torch.tensor(edge_df.iloc[:, :2].values.T, dtype=torch.long)
    edge_weight = torch.tensor(edge_df.iloc[:, 2].values, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)

    return data, node_structures


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



def get_motif_GCN_feature_dict (file_path: str): 
    data, node_structures = load_graph_from_excel(file_path)

    model = TwoLayerGCN(
        in_channels=data.num_node_features,
        hidden_channels=16,
        out_channels=8
    )

    model.eval()
    with torch.no_grad():
        node_embeddings = model(data)

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

    edges = list(graph.edges())

    edge_index = []

    for i, j in edges:

        u = Nodes_name_index_dict[i]
        v = Nodes_name_index_dict[j]
        edge_index.append((u, v))
        edge_index.append((v, u))

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
        atom_index = atom.GetIdx()  

        feature1_motif = motif_feature_dict[atom_to_motif[atom_index]]
        atom_id = str(num) + f'_atom_{atom_index}' 
        node_name.append(atom_id)
        topology_graph.add_node(atom_id)
        feature2_atom_en = torch.tensor(GetAtomFeature(atom, mol))

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



    for bond in mol.GetBonds():

        begin_atom_id = str(num) + f'_atom_{bond.GetBeginAtomIdx()}'
        end_atom_id = str(num) + f'_atom_{bond.GetEndAtomIdx()}'

        topology_graph.add_edge(end_atom_id, begin_atom_id)

    return node_name, node_features, topology_graph, Label_protect, Label_reacte, Label_pair_P_R



def Sample_creation(S_list,motif_feature_dict,reaction_dict,r):
    merged_graph = nx.Graph()
    Feature = defaultdict(dict)
    Label_Protect = []
    Label_Reacte = []
    Label_pair_Protect_Reacte = []
    name_list = []  
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


    df = pd.read_excel("Data/motif_embeddings.xlsx")

    motif_feature_dict = {
        row['label']: torch.tensor([float(x) for x in row['embedding'].split(",")], dtype=torch.float32)
        for _, row in df.iterrows()
    }

    df = pd.read_excel("Data/Data_reaction_atom_features_reactants.xlsx")


    reaction_dict = {}

    for _, row in df.iterrows():
        reaction = row["reaction"]
        reactant = row["reactant_smiles"]
        atom_idx = int(row["atom_idx"])


        key = f"{reaction}|{reactant}|{atom_idx}"

        feat_cols = [col for col in df.columns if col.startswith("feat_")]

        embedding = torch.tensor(row[feat_cols].astype(float).values, dtype=torch.float32)

        reaction_dict[key] = embedding


    Reaction_list = []
    with open(FilePath, 'r', encoding='UTF-8') as file_object:
        for line in file_object:
            Reaction_list.append(line)
    print('Number_Data:', len(Reaction_list))

    Datalist = []
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(motif_feature_dict)
    for i in range(len(Reaction_list)):
        r = Reaction_list[i]
        Reactants, Products = r.split('>>')
        Reactants_list = Reactants.split('.')
        Feature_index, edge_index_, Label_list = Sample_creation(Reactants_list,motif_feature_dict,reaction_dict,r)

        Fe = []
        for q in range(len(Feature_index)):
            Fe.append(Feature_index[q])

        F_tensor = F.normalize(torch.stack(Fe, dim=0), p=2, dim=0)
        data = Data(x=F_tensor, edge_index=edge_index_, y=torch.tensor(Label_list))

        Datalist.append(data)

    train_indices, test_indices = train_test_split(range(len(Datalist)), test_size=0.25, random_state=42)
    train_data_list = [Datalist[j] for j in train_indices]
    test_data_list = [Datalist[j] for j in test_indices]
    train_dataloader = DataLoader(train_data_list, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data_list, batch_size=1, shuffle=False)

    return train_dataloader, test_dataloader



