import numpy as np
import re
import rdkit
from rdkit import Chem
from rdkit.Chem import Mol, Draw
from rdkit.Chem.Lipinski import RotatableBondSmarts

'''
Some functions for filtering lists of (generative) molecules

functions:
FilterScore 		- filters molecules based on 'score'; either from RDKit fingerprint 
					  or an external method (e.g. Pytorch)
SortScores  		- takes a list of molecules and their scores and sorts from high-to-low
find_bond_groups 	- finds the longest sequence of rotatable bonds in conjunction with the 
					  ContiguousRotatableBonds function
FilterStructures	- filters structures by ring size (min-to-max), ring count (min-to-max) and rotors
SubstructureFilter  - searches a list of molecules for those that contain a given sub-structure
'''


def FilterScore(SmiArray,ScoreArray,Sort=True,Threshold=0.5,Range=0.05,silent=False,Unique=True):
    '''
    Simply filters an array of molecules based on some scores that are supplied.
    Scores are whatever metric you want to use. Combination of AM1 calcs and fingerprinting usually, but
    could be something build into rdkit.
    
    inputs:
    SmiArray   - molecules to be filtered, as smiles
    ScoreArray - their scores (from rdkit or G09 or pytorch), ideally in range 0-1
    Threshold  - the midpoint we'll score from (e.g. 0.5)
    Range      - how far we can deviate from threshold and not get filtered.
    
    outputs:
    Smi        - a list of smiles strings that were not filtered out.
    '''
    if not silent:
        print('\nFiltering for scores in range ' + str(Threshold-Range) + ':' + str(Threshold+Range))
    Smi = []
    Scores = []
    indices_thresh = [i for i,x in enumerate(ScoreArray) if x > Threshold-Range and x < Threshold+Range]
    Smi.append([(SmiArray[idx]) for idx in indices_thresh])
    Scores.append([(ScoreArray[idx]) for idx in indices_thresh])
    
    SmiList = [Item for Sublist in Smi for Item in Sublist]
    ScoresList = [Item for Sublist in Scores for Item in Sublist]
    
    if Unique:
        SmiList,ScoresList = FilterUnique(SmiList,ScoresList,silent=silent)
    
    if Sort:
        SmiList,ScoresList = SortScores(SmiList,ScoresList)
   
    if len(SmiList): # draw the rejects as a grid image
        if not silent:
            print('Total of ' + str(len(SmiList)) + ' molecules after filtering scores')
            display(Draw.MolsToGridImage([Chem.MolFromSmiles(x) for x in SmiList],
                                    molsPerRow=4, 
                                    legends=['Score: ' + str(np.round(Score,4)) for Score in ScoresList], 
                                    subImgSize=(400,300),
                                    maxMols=400))
            
    return(SmiList,ScoresList)

def FilterUnique(SmiList,ScoreList,silent=True):

    '''
    Simple function we can call to take a list of smiles strings and scores and return
    values only for unique smiles entries (identical scores are permitted but unlikely!)
    
    The order is preserved, so the nth smiles corresponds to the nth string.
    
    inputs:
    SmiList         - a list of smiles strings
    ScoresList      - a list of scores
    
    returns:
    UniqueSmi       - A list of unique smies strings
    UniqueScores    - A list of unique scores for above smiles list
    '''
    
    unique_smi_dict = {} # empty dict
    for smi,score in zip(SmiList, ScoreList):
        if smi not in unique_smi_dict:
            unique_smi_dict[smi] = score
        
    UniqueSmi = list(unique_smi_dict.keys())
    UniqueScores = [unique_smi_dict[smi] for smi in UniqueSmi]
    
    if not silent:
        print(str(len(SmiList)) + ' Input structures were filtered to ' + str(len(UniqueSmi)) + ' unique structures')
    
    return(UniqueSmi,UniqueScores)

def SortScores(SmiArray,ScoreArray):
    '''
    Sort an array of smiles strings by their scores, nowt more.
    
    inputs:
    SortedSmi is the sorted smiles strings
    SortedScore is the sorted scores
    
    returns:
    sortedsmi - smiles strings in score order
    sortedscore - scores in sorted order
    '''
    
    SortedSmi = [x for _, x in sorted(zip(ScoreArray, SmiArray))][::-1]
    SortedScore = sorted(ScoreArray)[::-1]
    
    return(SortedSmi,SortedScore)


def find_bond_groups(smi):
    '''
    Find groups of contiguous rotatable bonds and return them sorted by decreasing size
    Taken from RDKit cookbook (https://rdkit.org/docs/Cookbook.html) on 22/02/2023
    '''
    
    mol = Chem.MolFromSmiles(smi)
    rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
    rot_bond_set = set([mol.GetBondBetweenAtoms(*ap).GetIdx() for ap in rot_atom_pairs])
    rot_bond_groups = []
    while (rot_bond_set):
        i = rot_bond_set.pop()
        connected_bond_set = set([i])
        stack = [i]
        while (stack):
            i = stack.pop()
            b = mol.GetBondWithIdx(i)
            bonds = []
            for a in (b.GetBeginAtom(), b.GetEndAtom()):
                bonds.extend([b.GetIdx() for b in a.GetBonds() if (
                    (b.GetIdx() in rot_bond_set) and (not (b.GetIdx() in connected_bond_set)))])
            connected_bond_set.update(bonds)
            stack.extend(bonds)
        rot_bond_set.difference_update(connected_bond_set)
        rot_bond_groups.append(tuple(connected_bond_set))
    return tuple(sorted(rot_bond_groups, reverse = True, key = lambda x: len(x)))


def ContiguousRotatableBonds(smi):
    '''
    retrun the largest number of contiguous rotatable bonds for a given smiles string 
    Taken from RDKit cookbook (https://rdkit.org/docs/Cookbook.html) on 22/02/2023
    '''
    
    # Find groups of contiguous rotatable bonds in mol
    bond_groups = find_bond_groups(smi)
    # As bond groups are sorted by decreasing size, the size of the first group (if any)
    # is the largest number of contiguous rotatable bonds in mol
    largest_n_cont_rot_bonds = len(bond_groups[0]) if bond_groups else 0

    return (largest_n_cont_rot_bonds) 

def FilterStructures(smiles,MinRing=5,MaxRing=6,Rotors=3,CountCOO=False,MinRings=1,MaxRings=8):
    '''
    This function filters a list of smiles strings and filters out stuff we don't like. 
    Returns both hits and not-hits for the given filter.
    
    Inputs and defaults (in brackets)
    MinRing = minimum ring size allowed, in number of atoms (5)
    MaxRing = maximum ring size allowed, in number of atoms (7)
    Rotors = maximum number of contiguous rotatable bonds.  (3)
    MinRings = minimum number of rings (1)
    MaxRings = maximum number of rings (8)
    CountCOO = exact number of ester groups (42); note, if it equals 42 (!) then code will skip this loop.
    
    Returns:
    RetainList = structures that are not caught by any filters
    RejectList = structures caught by one or more filters.
    RejectLabel = a list of labels to use in the rdkit molstogridimage 'legend' so we can see why a given structure failed
    
    Future Ideas:
    * filter to remove antiaromatics, strained rings, brutally reactive things (e.g. *-O-O-O-*, *C#P?
    '''

    # Set up a list of Filters for ring size
    Filter = ['[r' + str(x) + ']' for x in range(3,MinRing)] + ['[r' + str(x) + ']' for x in range(MaxRing,20)]

    #Initialise our lists
    RetainList = []
    RejectList = []
    RejectLabel = []
    
    
    # loop over all molecules as smiles
    for smi in smiles:
        Mol = Chem.MolFromSmiles(smi) #convert to rdkit mol object
        
        #loop over all ring filters (r3,r4 etc.)
        for filter_str in Filter:
            
            #first up - fails both ring size and rotor count
            if  Mol.GetSubstructMatches(Chem.MolFromSmarts(filter_str)) and ContiguousRotatableBonds(smi)>= Rotors:
                RejectLabel.append('Ring Size Violation and\nContiguous Rotors Violation (' + str(ContiguousRotatableBonds(smi))+')')
                RejectList.append(Mol)
                break
                
            #look for fails of rotor count only    
            if ContiguousRotatableBonds(smi)>=Rotors:
                RejectLabel.append('Contiguous Rotors Violation (' + str(ContiguousRotatableBonds(smi))+')')
                RejectList.append(Mol)
                break
            
            #look for fail of ring size only
            if Mol.GetSubstructMatches(Chem.MolFromSmarts(filter_str)):
                RejectLabel.append('Ring Size Violation')
                RejectList.append(Mol)
                break
            
            # look for wrong number of esters (COO)
            if CountCOO:
                if Mol.GetSubstructMatches(Chem.MolFromSmarts('[#6](=[#8])-[#8]')) != CountCOO:
                    RejectLabel.append('Carboxylate Ester Violation')
                    RejectList.append(Mol)
                    break
                    
            # look for wrong number of rings
            if MaxRings < len(Mol.GetRingInfo().AtomRings()) or  len(Mol.GetRingInfo().AtomRings()) < MinRings:
                RejectLabel.append('Ring Count Violation ' + str(len(Mol.GetRingInfo().AtomRings())))
                RejectList.append(Mol)
                break
                
            # lastly if it passes, add it to the retain list
            if not Mol.GetSubstructMatches(Chem.MolFromSmarts(filter_str)) and ContiguousRotatableBonds(smi)< Rotors:
                RetainList.append(Mol)
                break
                
    # give a little feedback to the user
    print('Total of ' + str(len(smiles)) + ' molecules initially')
    print('Total of ' + str(len(RejectList)) + ' molecules were filtered out')
    print('Total of ' + str(len(RetainList)) + ' molecules were not filtered out')
    
    if RejectList: # draw the rejects as a grid image
        display(Draw.MolsToGridImage(RejectList,legends=RejectLabel,molsPerRow=4,subImgSize=(400,300),maxMols=400))
    
    return(RetainList,RejectList,RejectLabel)


def SubstructureFilter(smiles,substructure):
    '''
    a **crude** method for filtering a list of smiles strings for a the occurance
    of a given sub-structure.
    
    Returns the list of hits, the list of not-hits, and the number of hits.
    '''
    # set up lists:
    RetainList=[]
    RejectList=[]
    
    for smi in smiles:
        Mol = Chem.MolFromSmiles(smi)
        
        if Mol.GetSubstructMatches(Chem.MolFromSmiles(substructure)):
            RetainList.append(Mol)
            
        if not Mol.GetSubstructMatches(Chem.MolFromSmiles(substructure)):
            RejectList.append(Mol) 
    
    print('I looked at ' + str(len(smiles)) + ' molecules to find those with the substructure: ' + substructure)
    
    im = Chem.Draw.MolToImage(Chem.MolFromSmiles(substructure))
    
    print('Total of ' + str(len(RejectList)) + ' molecules have no substructure match')
    print('Total of ' + str(len(RetainList)) + ' molecules have a substructure match')
    
    if RetainList: # draw the hits as a grid image
        display(Draw.MolsToGridImage(RetainList,molsPerRow=4,subImgSize=(400,300),maxMols=400))
        
    return(RetainList,RejectList)
    
def SaveSmilesScores(SmiList,ScoresList,filename="SmilesAndScores"):
    '''
    Little one-liner for saving data to a .csv file; we'll want smiles and scores I guess
    '''

    if filename[-4:] != '.csv': #give it the right name
        filename = filename+'.csv'

    np.savetxt(filename,np.array([SmiList,ScoresList]),delimiter=", ",fmt ='% s')
    print('saved structures and scores as ' + filename)
    return