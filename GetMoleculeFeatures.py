from rdkit import Chem

Symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn','H', 'Cu', 'Mn', '*','unknown'
    ,'SINGLE','DOUBLE','TRIPLE','AROMATIC']
Symbol_One_Hot = {}
for i in range(len(Symbols)):
    Symbol_One_Hot[Symbols[i]] = [0]*i + [1] + [0]*(len(Symbols)-i-1)

BondEnergy_Table = {}
with open('Data/bond_energy.txt', 'r', encoding='UTF-8') as file_object:
    for line in file_object:
        line = line.rstrip()
        BondName, Energy = line.split('	')
        BondEnergy_Table[BondName] = int(Energy)


AtomQuality_Table = {}
with open('Data/atom_mass.txt', 'r',encoding='UTF-8') as file_object:
    for line in file_object:
        line = line.rstrip()
        AtomName, Quality = line.split('	')
        AtomQuality_Table[AtomName] = float(Quality)


Atom_electronegativity_Table = {}
with open('Data/atom_en.txt', 'r', encoding='UTF-8') as file_object:
    for line in file_object:
        line = line.rstrip()
        AtomName, electronegativity = line.split('	')
        Atom_electronegativity_Table[AtomName] = float(electronegativity)



def GetAtomFeature(Atom_Rdkit,mol):
    Feature = []
    Atom_Symbol = Atom_Rdkit.GetSymbol()
    Degree = Atom_Rdkit.GetDegree() 
    IsAromatic = Atom_Rdkit.GetIsAromatic()  
    TotalNumHs = Atom_Rdkit.GetTotalNumHs()  
    fc = Atom_Rdkit.GetFormalCharge()  

    if Atom_Symbol not in Symbol_One_Hot.keys():
        Atom_Symbol = 'unknown'
    if str(IsAromatic) == 'False':
        IsAromatic = 0
    elif str(IsAromatic) == 'True':
        IsAromatic = 1
    Feature += Symbol_One_Hot[Atom_Symbol]  
    Feature.append(TotalNumHs) 
    Feature.append(Degree)  
    Feature.append(IsAromatic)  
    Feature.append(fc)  
    Feature.append(AtomQuality_Table[Atom_Symbol])  
    Feature.append(Atom_electronegativity_Table[Atom_Symbol])  
 
    return Feature


def test(Atom_Rdkit):
    Feature = []
    Atom_Symbol = Atom_Rdkit.GetSymbol()
    Degree = Atom_Rdkit.GetDegree()  
    IsAromatic = Atom_Rdkit.GetIsAromatic()  
    TotalNumHs = Atom_Rdkit.GetTotalNumHs() 
    fc = Atom_Rdkit.GetFormalCharge()  
    if Atom_Symbol not in Symbol_One_Hot.keys():
        Atom_Symbol = 'unknown'
    if str(IsAromatic) == 'False':
        IsAromatic = 0
    elif str(IsAromatic) == 'True':
        IsAromatic = 1
    Feature += Symbol_One_Hot[Atom_Symbol]  
    Feature.append(TotalNumHs)  
    Feature.append(Degree)  
    Feature.append(IsAromatic)  
    Feature.append(fc)  
    Feature.append(AtomQuality_Table[Atom_Symbol]) 
    Feature.append(Atom_electronegativity_Table[Atom_Symbol]) 
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
        print('Others')
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

    length = len(test(begin_atom))

    Feature += [0] * (length-len(Feature))
    Feature.append(BondEnergy)

    return Feature








