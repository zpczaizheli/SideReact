from rdkit import Chem
'''注意：如果要更改或者添加原子的初始特征，一定注意更改多个特征提取函数，保持一致性'''


Symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn','H', 'Cu', 'Mn', '*','unknown'
    ,'SINGLE','DOUBLE','TRIPLE','AROMATIC']
Symbol_One_Hot = {}
for i in range(len(Symbols)):
    Symbol_One_Hot[Symbols[i]] = [0]*i + [1] + [0]*(len(Symbols)-i-1)

BondEnergy_Table = {}
with open('Data/键能表.txt', 'r', encoding='UTF-8') as file_object:
    for line in file_object:
        line = line.rstrip()
        BondName, Energy = line.split('	')
        BondEnergy_Table[BondName] = int(Energy)


AtomQuality_Table = {}
with open('Data/原子质量表.txt', 'r',encoding='UTF-8') as file_object:
    for line in file_object:
        line = line.rstrip()
        AtomName, Quality = line.split('	')
        AtomQuality_Table[AtomName] = float(Quality)


Atom_electronegativity_Table = {}
with open('Data/原子电负性表.txt', 'r', encoding='UTF-8') as file_object:
    for line in file_object:
        line = line.rstrip()
        AtomName, electronegativity = line.split('	')
        Atom_electronegativity_Table[AtomName] = float(electronegativity)



def GetAtomFeature(Atom_Rdkit,mol):
    Feature = []
    Atom_Symbol = Atom_Rdkit.GetSymbol()
    Degree = Atom_Rdkit.GetDegree()  # 该原子的度
    IsAromatic = Atom_Rdkit.GetIsAromatic()  # 是否是芳香环
    TotalNumHs = Atom_Rdkit.GetTotalNumHs()  # 与该原子连接的氢原子个数
    fc = Atom_Rdkit.GetFormalCharge()  # 该原子所带电荷数(1维)

    if Atom_Symbol not in Symbol_One_Hot.keys():
        Atom_Symbol = 'unknown'
    if str(IsAromatic) == 'False':
        IsAromatic = 0
    elif str(IsAromatic) == 'True':
        IsAromatic = 1
    Feature += Symbol_One_Hot[Atom_Symbol]  # 用one-hot编码表示原子的符号 (23维)
    Feature.append(TotalNumHs)  # 与该原子连接的氢原子个数(1维)
    Feature.append(Degree)  # 该原子的度(1维)
    Feature.append(IsAromatic)  # 该原子是否是芳香环上的原子(1维)
    Feature.append(fc)  # 该原子所带电荷数(1维)
    Feature.append(AtomQuality_Table[Atom_Symbol])  # 该原子的原子质量(1维)
    Feature.append(Atom_electronegativity_Table[Atom_Symbol])  # 该原子的电负性（1维）
    # bond = mol.GetBondBetweenAtoms(0, 1)
    # length = len(GetBondFeature(bond))
    # Feature += [0] * (length-len(Feature))
    return Feature


def test(Atom_Rdkit):
    Feature = []
    Atom_Symbol = Atom_Rdkit.GetSymbol()
    Degree = Atom_Rdkit.GetDegree()  # 该原子的度
    IsAromatic = Atom_Rdkit.GetIsAromatic()  # 是否是芳香环
    TotalNumHs = Atom_Rdkit.GetTotalNumHs()  # 与该原子连接的氢原子个数
    fc = Atom_Rdkit.GetFormalCharge()  # 该原子所带电荷数(1维)
    if Atom_Symbol not in Symbol_One_Hot.keys():
        Atom_Symbol = 'unknown'
    if str(IsAromatic) == 'False':
        IsAromatic = 0
    elif str(IsAromatic) == 'True':
        IsAromatic = 1
    Feature += Symbol_One_Hot[Atom_Symbol]  # 用one-hot编码表示原子的符号 (23维)
    Feature.append(TotalNumHs)  # 与该原子连接的氢原子个数(1维)
    Feature.append(Degree)  # 该原子的度(1维)
    Feature.append(IsAromatic)  # 该原子是否是芳香环上的原子(1维)
    Feature.append(fc)  # 该原子所带电荷数(1维)
    Feature.append(AtomQuality_Table[Atom_Symbol])  # 该原子的原子质量(1维)
    Feature.append(Atom_electronegativity_Table[Atom_Symbol])  # 该原子的电负性（1维）
    return Feature


def GetBondFeature(bond):
    Feature = []
    BondType = bond.GetBondType()
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    BondSymbol = ''
    if str(BondType) == 'SINGLE':
        BondSymbol = '-'
    elif str(BondType) == 'DOUBLE':
        BondSymbol = '='
    elif str(BondType) == 'TRIPLE':
        BondSymbol = '#'
    elif str(BondType) == 'AROMATIC':
        BondSymbol = '~'
    else:
        print('其他键类型')
    if begin_atom == '*' or end_atom == '*':
        BondEnergy_Table[begin_atom + BondSymbol + end_atom] = 0
        BondEnergy_Table[end_atom + BondSymbol + begin_atom] = 0
    Bond1 = begin_atom.GetSymbol() + BondSymbol + end_atom.GetSymbol()
    Bond2 = end_atom.GetSymbol() + BondSymbol + begin_atom.GetSymbol()
    if Bond1 in BondEnergy_Table.keys():
        BondEnergy = BondEnergy_Table[Bond1]
    elif Bond2 in BondEnergy_Table.keys():
        BondEnergy = BondEnergy_Table[Bond2]
    else:
        BondEnergy = 0

    Feature += Symbol_One_Hot[str(BondType)]
    # 维度补全
    length = len(test(begin_atom))
    # print(len(Feature))
    Feature += [0] * (length-len(Feature))
    Feature.append(BondEnergy)
    # print(length)
    return Feature








