import rdkit.Chem as Chem
import numpy as np
from tqdm import tqdm
import random

'''
This script prepares the data used in Wengong Jin's NIPS paper on predicting reaction outcomes for the modified
forward prediction script. Rather than just training to predict which bonds change, we make a direct prediction
on HOW those bonds change
'''

def get_changed_bonds(rxn_smi):
    rxn_smi=rxn_smi.replace(' ','')

    reactants = Chem.MolFromSmiles(rxn_smi.split('>')[0])
    products = Chem.MolFromSmiles(rxn_smi.split('>')[2])


    conserved_maps = [a.GetProp('molAtomMapNumber') for a in products.GetAtoms() if a.HasProp('molAtomMapNumber')]
    bond_changes = set() # keep track of bond changes

    # Look at changed bonds
    bonds_prev = {}
    for bond in reactants.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        if (nums[0] not in conserved_maps) and (nums[1] not in conserved_maps): continue
        bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    bonds_new = {}
    for bond in products.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()


    for bond in bonds_prev:
        if bond not in bonds_new:
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], 0.0)) # lost bond
        else:
            if bonds_prev[bond] != bonds_new[bond]:
                bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond])) # changed bond
    for bond in bonds_new:
        if bond not in bonds_prev:
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))  # new bond

    return bond_changes


def process_file(fpath):
    with open(fpath, 'r') as fid_in:
        for line in tqdm(fid_in):
            rxn_smi = line.strip().split(' ')[0]
            # x=random.uniform(0,1)
            # if(x<0.57):
            #     fid_out.write('{}\n'.format(rxn_smi))
                # print(rxn_smi)
            bond_changes = get_changed_bonds(rxn_smi)
            # fid_out.write('{} {}\n'.format(rxn_smi, ';'.join(['{}-{}-{}'.format(x[0], x[1], x[2]) for x in bond_changes])))
            print('{} {}\n'.format(rxn_smi, ';'.join(['{}-{}-{}'.format(x[0], x[1], x[2]) for x in bond_changes])))
            # else:
            #     bond_changes = get_changed_bonds(rxn_smi)
            #     fid_out_ul.write('{} {}\n'.format(rxn_smi, ';'.join(['{}-{}-{}'.format(x[0], x[1], x[2]) for x in bond_changes])))
            #     fid_out_ul.write('{}\n'.format(rxn_smi.split('>')[0]))
                # print('{}\n'.format(rxn_smi.split('>')[0]))
    print('Finished processing {}'.format(fpath))

def find_same(path1,path2):
    with open(path1,'r') as file1,open(path2,'r') as file2:

        unlabel=[]
        for line1 in tqdm(file1):
            unlabel.append(line1)
        print('unlabel set end')
        for line2 in tqdm(file2):
            # print(line2)
            reac=line2.split('>')[0]
            # print(reac)
            if reac in unlabel:
                print('have same')
                break
    print('end')

def process(path):
    with open(path,'r') as read, open(path+'.proc','w') as out:
        for line in tqdm(read):
            mol=Chem.MolFromSmiles(line.replace(' ',''))
            atoms=mol.GetNumAtoms()
            for idx in range(atoms):
                mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()+1))
            smi=Chem.MolToSmiles(mol)
            # print('{}\n'.format(smi))
            out.write('{}\n'.format(smi))

def merge(path1,path2):
    with open(path1,'r') as read1, open(path2,'r') as read2, open('../data/withproduct.txt','w') as out:
        for line in tqdm(read1):
            line1=read2.readline()
            # print('{}>>{}'.format(line.strip().split('\n')[0],line1))
            out.write('{}>>{}'.format(line.strip().split('\n')[0],line1))


if __name__ == '__main__':
    # Test summarization
    # for rxn_smi in [
    #         '[CH2:15]([CH:16]([CH3:17])[CH3:18])[Mg+:19].[CH2:20]1[O:21][CH2:22][CH2:23][CH2:24]1.[Cl-:14].[OH:1][c:2]1[n:3][cH:4][c:5]([C:6](=[O:7])[N:8]([O:9][CH3:10])[CH3:11])[cH:12][cH:13]1>>[OH:1][c:2]1[n:3][cH:4][c:5]([C:6](=[O:7])[CH2:15][CH:16]([CH3:17])[CH3:18])[cH:12][cH:13]1',
    #         '[CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]',
    #         '[CH2:1]([CH3:2])[n:3]1[cH:4][c:5]([C:22](=[O:23])[OH:24])[c:6](=[O:21])[c:7]2[cH:8][c:9]([F:20])[c:10](-[c:13]3[cH:14][cH:15][c:16]([NH2:19])[cH:17][cH:18]3)[cH:11][c:12]12.[CH:25](=[O:26])[OH:27]>>[CH2:1]([CH3:2])[n:3]1[cH:4][c:5]([C:22](=[O:23])[OH:24])[c:6](=[O:21])[c:7]2[cH:8][c:9]([F:20])[c:10](-[c:13]3[cH:14][cH:15][c:16]([NH:19][CH:25]=[O:26])[cH:17][cH:18]3)[cH:11][c:12]12',
    #         ]:
    #     print(rxn_smi)
    #     print(get_changed_bonds(rxn_smi))

    # Process files
    # process_file('../data/train.txt')
    # process_file('../data/valid.txt')
    process_file('../data/test.txt')
    # process_file('../data/test_human.txt')
    # process_file('../data/precursors-train.txt.proc')
    # process_file('../data/withproduct.txt')
    # process('../data/product-train.txt')
    # merge('../data/precursors-train.txt.proc','../data/product-train.txt.proc')
    # find_same('../data/precursors-train.txt.proc','../data/train.txt.proc')