from rdkit import Chem
# from rdkit.Chem import rdmolops, rdMolFragmenter
import pandas as pd
from rdkit.Chem import Descriptors
from typing import Dict
import pandas as pd
from rdkit.Chem import Draw
def assign_origin_index(mol):
    for atom in mol.GetAtoms():
        atom.SetIntProp('origin_index', atom.GetIdx())
    return mol


def match_fragment_to_motifs(fragment, motif_list):
    count = 0
    for motif in motif_list:
        mol = Chem.MolFromSmarts(motif)
        # print(fragment,'----',mol)
        if fragment.HasSubstructMatch(mol) and mol.HasSubstructMatch(fragment):
            return motif
    return "xxxxx"

def remove_dummy_atoms(mol):
    dummy = Chem.MolFromSmiles('*')
    mol = Chem.DeleteSubstructs(mol, dummy)
    sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
    return mol



def split_molecule_by_rotatable_bonds(mol):

    # 将 SMILES 转换为分子对象
    # mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的 SMILES 字符串！")

    # 找到可旋转键的位置
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    rotatable_bonds = []
    for bond in mol.GetBonds():
        if bond.IsInRing():  # 排除环内键
            continue
        if bond.GetBondType() == Chem.BondType.SINGLE:  # 仅考虑单键
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            # 判断是否为末端原子（连接的键数是否为 1）
            if begin_atom.GetDegree() >= 1 and end_atom.GetDegree() >= 1:
                rotatable_bonds.append(bond.GetIdx())

    if not rotatable_bonds:
        return [Chem.MolToSmiles(mol)]  # 如果没有可旋转键，直接返回原分子

    # 标记可旋转键并断键
    fragments = Chem.FragmentOnBonds(mol, rotatable_bonds)
    frags = Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=True)
    # print('1',Chem.MolToSmiles(frags[3]))

    clean_frags = []
    for frag in frags:
        clean_frag = remove_dummy_atoms(frag)
        clean_frags.append(clean_frag)
    # print('2',Chem.MolToSmiles(clean_frags[3]))

    # atom = clean_frags[0].GetAtomWithIdx(0)
    # custom_value = atom.GetIntProp('origin_index')
    # print('origin_index:', custom_value)
    return clean_frags


def generate_mapping_table(target_mol, motif_list) -> Dict[int, Chem.Mol]:
    assign_origin_index(target_mol)
    fragments = split_molecule_by_rotatable_bonds(target_mol)

    mapping = {}
    # print(motif_list)
    for frag in fragments:
        matched_motif = match_fragment_to_motifs(frag, motif_list)
        # print(type(matched_motif))
        for atom in frag.GetAtoms():
            if atom.HasProp('origin_index'):
                origin_index = atom.GetIntProp('origin_index')
                mapping[origin_index] = matched_motif

    return mapping

# if __name__ == "__main__":
#     # Example usage
#     motif_smiles_list = ["C1=CC=CC=C1",
#                          "CCO",
#                          "CC",
#                          "C",
#                          ]
#     motif_list = [Chem.MolFromSmiles(smiles) for smiles in motif_smiles_list]
#
#     target_smiles = "c1ccccc1CCC"
#     target_mol = Chem.MolFromSmiles(target_smiles)
#
#     mapping_table = generate_mapping_table(target_mol, motif_list)
#     print(mapping_table)