import rdkit
import rdkit.Chem as Chem
import numpy as np
import random

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 10

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + [atom.GetIsAromatic()], dtype=np.float32)

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)

def smiles2graph(smiles, idxfunc=lambda x:x.GetIdx()):
    # smiles=smiles.replace(' ','')
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse smiles string:", smiles)

    n_atoms = mol.GetNumAtoms()
    n_bonds = max(mol.GetNumBonds(), 1)
    fatoms = np.zeros((n_atoms, atom_fdim))
    fatoms_p = np.zeros((n_atoms, atom_fdim))    #扰动
    fbonds = np.zeros((n_bonds, bond_fdim))
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)
    degree = np.eye(n_atoms) # 增加了度矩阵

    # print(len(mol.GetAtoms()))
    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        if idx >= n_atoms:
            raise Exception(smiles)
        i=random.random()
        if i>0.15:
            fatoms_p[idx] = atom_features(atom)
        fatoms[idx] = atom_features(atom)

    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()
        if num_nbs[a1] == max_nb or num_nbs[a2] == max_nb:
            raise Exception(smiles)
        atom_nb[a1,num_nbs[a1]] = a2
        atom_nb[a2,num_nbs[a2]] = a1
        bond_nb[a1,num_nbs[a1]] = idx
        bond_nb[a2,num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1
        degree[a1, a1] += 1
        degree[a2, a2] += 1
        fbonds[idx] = bond_features(bond)
    row,col=np.diag_indices_from(degree)
    degree[row,col]=degree[row,col]**-1
    return fatoms, fatoms_p, fbonds, atom_nb, bond_nb, num_nbs, degree

def pack2D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m] = arr
    return a

def pack2D_withidx(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M, 2))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m,0] = i
        a[i,0:n,0:m,1] = arr
    return a

def pack1D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        a[i,0:n] = arr
    return a

def get_mask(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        for j in range(arr.shape[0]):
            a[i][j] = 1
    return a

def smiles2graph_list(smiles_list, idxfunc=lambda x:x.GetIdx()):
    '''
    This function prepares all of the model inputs needed to process one batch and
    pads them as needed (because not all examples will have the same number of atoms)
    '''
    res = list(map(lambda x:smiles2graph(x,idxfunc), smiles_list))
    fatom_list, fatom_p_list, fbond_list, gatom_list, gbond_list, nb_list, de_list= zip(*res)
    return pack2D(fatom_list), pack2D(fatom_p_list),pack2D(fbond_list), pack2D_withidx(gatom_list), pack2D_withidx(gbond_list), pack1D(nb_list), pack2D(de_list), get_mask(fatom_list)

if __name__ == "__main__":
    # np.set_printoptions(threshold='nan')
    a,b,c,d,e,f,g,h = smiles2graph_list(['C C O P ( = O ) ( C O ) O C C . C N ( C ) C = O . N c 1 n c n c 2 c 1 n c ( C Br ) n 2 C C c 1 c c c c c 1 . [Na]'])
    print(a)
    # print('------------------')
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)
    print(g)
    print(h)
