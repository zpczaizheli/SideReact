from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from rdkit import Chem

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('pretrain_model/Qwen2___5-0___5B-Instruct', trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    '50k_model/2025042119/checkpoint-200000',
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

print('Loading dataset')
df = pd.read_csv('data/test-2.csv')
ds = Dataset.from_pandas(df)
ds = ds.shuffle()

# system prompt (略)
p = '输入反应式，输出反应类别[1-10]...'


def get_atom_token_mapping(smiles, tokenizer, model):
    """
    给定一个 SMILES，返回 {atom_idx: (symbol, embedding向量)}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    # 用 canonical smiles 确保 RDKit 原子顺序和字符串顺序一致
    can_smiles = Chem.MolToSmiles(mol, canonical=True)

    # Tokenizer 带 offset_mapping
    enc = tokenizer(can_smiles, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    input_ids = enc["input_ids"].to('cuda')   # 已经是 [1, seq_len]

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states[-1].squeeze(0).float().cpu()  # [seq_len, hidden_dim]

    # 建立原子→字符位置映射
    atom_pos = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        pos = can_smiles.find(symbol)
        if pos == -1:
            continue
        atom_pos[idx] = (symbol, pos, pos + len(symbol))

    # 匹配 token offset 到原子
    atom_embeddings = {}
    for idx, (sym, start, end) in atom_pos.items():
        token_ids = [ti for ti, (s, e) in enumerate(offsets) if not (e <= start or s >= end)]
        if not token_ids:
            continue
        vecs = [hs[ti] for ti in token_ids]
        atom_embeddings[idx] = (sym, torch.stack(vecs, dim=0).mean(dim=0).tolist())

    return atom_embeddings


records = []
for i in tqdm(range(len(ds)), desc="Processing"):
    rxn_smiles = ds[i]['rxn_smiles']
    try:
        reactants, products = rxn_smiles.split(">>")
    except:
        continue

    # 多个反应物，按 "." 分开
    reactant_mols = reactants.split(".")

    for rid, r in enumerate(reactant_mols):  
        reactant_feats = get_atom_token_mapping(r, tokenizer, model)

        # 给当前反应物单独编号，从0开始
        local_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(reactant_feats.keys())}

        for old_idx, (sym, feat) in reactant_feats.items():
            local_idx = local_mapping[old_idx]
            # 这里多加一列 reactant_smiles
            records.append([rxn_smiles, f"reactant_{rid}", r, sym, local_idx] + feat)

# 保存
hidden_size = len(records[0]) - 5  # 注意现在多了一列 reactant_smiles
columns = ["reaction", "role", "reactant_smiles", "atom_symbol", "atom_idx"] + [f"feat_{i}" for i in range(hidden_size)]
df_features = pd.DataFrame(records, columns=columns)
df_features.to_excel("reaction_atom_features_reactants.xlsx", index=False, engine="openpyxl")

print("Reactant-only atom-level features (local numbering + reactant SMILES) saved to reaction_atom_features_reactants.xlsx")


