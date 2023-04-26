import random
import tqdm
import re
import numpy as np
import itertools

from group_selfies import (
    fragment_mols, 
    Group, 
    MolecularGraph, 
    GroupGrammar, 
    group_encoder
)

from rdkit import Chem
from rdkit.Chem import Mol, Draw

'''
Functions for generative molecular deisgn via Group-Selfies

Most of the backend is in this paper

and the code is here
'''

def ParseFragments(Subset = 2500,Method='default',Display=True):
    '''
    Get some molecules from the .csv database and break a subset into fragments
    '''
    
    if Method != 'default' and Method != 'mmpa':
        Method = 'default'
    
    LCFrags = [x.strip() for x in open('./LCFrags.csv')]
    subset = random.sample(LCFrags, Subset)
    
    if Method == 'default':
        fragments = fragment_mols(subset, convert=True, method='default') # use custom fragmentation technique
        vocab_fragment = dict([(f'frag{idx}', Group(f'frag{idx}', frag)) for idx, frag in enumerate(fragments)])
        if Display == True:
            display(Draw.MolsToGridImage([g.mol for g in vocab_fragment.values()]))
        return(vocab_fragment)
        
    if Method == 'mmpa':
        print('using MMPA')
        fragments_mmpa = fragment_mols(subset, convert=True, method='mmpa') # use MMPA fragmentation
        vocab_fragment = dict([(f'frag{idx}', Group(f'frag{idx}', frag)) for idx, frag in enumerate(fragments_mmpa)])
        
        if Display == True:
            display(Draw.MolsToGridImage([g.mol for g in vocab_fragment.values()]))
        return(vocab_fragment)


def DefinedFragments():
    '''
    Rather than use a database of known/guessed molecules (with ParseFragments) to generate our 
    fragments, we can simply look them up.
    
    This list was built with a search of Reaxys for known LC materials which were then converted to
    smiles and sorted into substrings, e.g. phenyl, pyrmidine, ester, cyano etc.
    
    Just call the function in the main notebook and it will pass the grammar back.
    
    You can add new entries here if you want.
    
    
    '''
    
    # First, build our list of fragments up. 
    # we split by Rings and Groups. Rings are allowed to have attachments at any atoms, groups are not.
    # there are subdivisions into aromatics, heterocycles, link units etc.
    
    Rings = [
    # Rings only; all attachment points allowed (generally)
    # To avoid hypervalency we'll specify the number of attachments for S-containing heterocycles.
    # aromatics
    ['Benzene','c0ccccc0'],
    ['Resorcinol','c0c(O)cc(O)cc0'],
    ['Catechol','c0c(O)c(O)ccc0'],
    ['Hydroquinone','c0c(O)ccc(O)c0'],
    ['Resorcinol dimethylether','c0c(OC)cc(OC)cc0'],
    ['Catechol dimethylether','c0c(OC)c(OC)ccc0'],
    ['Hydroquinone dimethylether','c0c(OC)ccc(OC)c0'],
    ['Fluorobenzene','c0c(F)cccc0'],
    ['Difluorobenzene(2,3)','c0c(F)c(F)ccc0'],
    ['Difluorobenzene(3,5)','c0c(F)ccc(F)c0'],
    ['Difluorobenzene(3,5)','c0cc(F)cc(F)c0'],
    ['Trifluorobenzene(3,4,5)','c0cc(F)c(F)c(F)c0'],
    ['Pyridine','c0cnccc0'],
    ['Pyrimidine','c0cncnc0'],
    ['Pyrazine','c0nccnc0'],
    ['Pyrazidine','c0nnccc0'],
    ['Naphthalene','c1c2ccccc2ccc1'],
    ['Indene','c1ccc2c(c1)CC=C2'],
    ['Azulene','c0ccc1cccc1cc0'],
    ['Fluorene','c0ccc1c2ccccc2Cc1c0'],
    ['Fluorenone','c0ccc1c2ccccc2C(=O)c1c0'],
    ['Indane','c0ccc1CCCc1c0'],
    ['Anthracene','c0ccc1cc2ccccc2cc1c0'],
   
    #Heteroaromatics
    # Rdkit (or maybe selfies?) likes to have more than 2 bonds at S/Se; obviously
    # this is legit chemistry-wise, but in the present case we limit the connectivity
    # here with a (*x) flag, where the total number of bonds is valence-x.
    
    ['Indole','c0cccc1cc[nH]c01'],
    ['Furan','c0occc0'],
    ['Thiophene','c0[S](*3)(*1)ccc0'],
    ['Selenophene','c0[Se](*3)(*1)ccc0'],
    ['Benzofuran','c0cccc1[o]ccc01'],
    ['Benzothiophene','c0cccc1[s](*3)(*1)ccc01'],
    ['Oxadiazole(1,3,4)','c0nnco0'],
    ['Oxadiazole(1,2,5)','o0nccn0'],
    ['Oxadiazole(1,2,3)','o0ccnn0'],
    ['Thiadiazole(1,3,4)','c0nnc[s](*3)(*1)0'],
    ['Thiadiazole(1,2,5)','[s](*3)(*1)0nccn0'],
    ['Thiadiazole(1,2,3)','[s](*3)(*1)0ccnn0'],
    ['Pyrroline(3)','N0CC=CC0'],
    ['Pyrroline(2)','N0C=CCC0'],
    ['Pyrrole(2)','N0=CC=CC0'],
    ['Pyrrole(1)','N0C=CC=C0'],
    ['Pyrazole','N0N=CC=C0'],
    ['Imidazole','N0C=NC=C0'],
    ['Triazole(1,2,4)','N0N=CN=C0'],
    ['Triazole(1,2,3)','N0N=NC=C0'],
    ['Tetrazole','N0N=NN=C0'],
    ['Oxazole','O0C=NC=C0'],
    ['Isoxazole','O0N=CC=C0'],
    ['Isothiazole','S0N=CC=C0'],
    ['Thiazole','[S](*3)(*1)0C=NC=C0'],
    
    
    #unsaturated
    ['Cyclohexene','C0=CCCCC0'],
    ['Cyclohexadiene','C0=CC=CCC0'],
    
    #saturated
    #simple
    ['Cyclopropane','C0CC0'],
    ['Cyclobutane','C0CCC0'],
    ['Cyclopentane','C0CCCC0'],
    ['Cyclohexane','C0CCCCC0'],
    ['Cycloheptane','C0CCCCCC0'],
    ['Cyclooctane','C0CCCCCCC0'],
    ['Cyclononane','C0CCCCCCCC0'],
    ['Cyclodecane','C0CCCCCCCCC0'],
    
    #heterocyclic
    ['Dioxane','C0OCCCO0'],
    ['Pyran','C0CCCCO0'],
    ['Dioxaborolane','B0OCCCO0'],
    ['Dithiane(1,3)','C0[S](*3)(*1)CCC[S](*3)(*1)0'],  #this does not work; gets fragmented to thiols etc.
    ['Oxythiane(1,3)','C0[S](*3)(*1)CCCO0'], #this also does not work; gets fragmented to thiols etc.
    ['Thietane','C0[S](*3)(*1)CC0'],         #this ALSO does not work; gets fragmented to thiols etc.
    ['Oxetane','C0OCC0'],
    ['Oxirane','C0OC0'],
    
    #bicyclo
    #note; a lot of the bicyclo- and complex- things are not easy to synthesise... trisubstituted cubanes :S
    # we are going to restrict the bicycloalkanes to bonding at bridgehead ONLY for all but BCA<1.1.1>pentane
    ['Bicyclo<1.1.1>pentane','C01CC(C0)C1'],
    ['Bicyclo<1.1.2>hexane','C01C(*2)C(C(*2)0)C(*2)C(*2)1'],
    ['Bicyclo<2.1.1>hexane','C01C(*2)C(*2)C(C(*2)0)C(*2)1'],
    ['Bicyclo<3.1.1>heptane','C01C(*2)C(*2)C(*2)C(C(*2)0)C(*2)1'],
    ['Bicyclo<2.2.1>heptane','C01C(*2)C(*2)C(C(*2)C(*2)0)C(*2)1'],
    ['Bicyclo<3.2.1>octane','C01C(*2)C(*2)C(*2)C(C(*2)C(*2)0)C(*2)1'],
    ['Bicyclo<2.2.2>octane','C01C(*2)C(*2)C(C(*2)C(*2)0)C(*2)C(*2)1'],
    ['Bicyclo<3.3.1>nonane','C01C(*2)C(*2)C(*2)C(C(*2)C(*2)C(*2)0)C(*2)1'],
    ['Bicyclo<3.2.2>nonane','C01C(*2)C(*2)C(*2)C(C(*2)C(*2)0)C(*2)C(*2)1'],
    ['Bicyclo<3.3.2>decane','C01C(*2)C(*2)C(*2)C(C(*2)C(*2)C(*2)0)C(*2)C(*2)1'],
    ['Bicyclo<3.2.3>decane','C01C(*2)C(*2)C(*2)C(C(*2)C(*2)0)C(*2)C(*2)C(*2)1'],
    ['Bicyclo<3.3.3>undecane','C01C(*2)C(*2)C(*2)C(C(*2)C(*2)C(*2)0)C(*2)C(*2)C(*2)1'],
    
    #complex
    ['cubane','C01C4C2C0C3C2C4C31'],
    ['Triphenylene','c0c(cc1c(c0)c2cc(c(cc2c3cc(c(cc13)))))'],
    ['Truxene','c0cc1Cc2c5c6ccccc6Cc5c3c4ccccc4Cc3c2c1cc0'],
    ]

    Groups =[#Groups doesn't include rings!
    #terminal stuff
    ['NO2','*1[N+](=O)[O-]'],
    ['CN','*1C#N'],
    ['NCS','*1N=C=S'],
    ['SF5','*1S(F)(F)(F)(F)(F)'],
    ['NH2','*1N'],
    ['SO2F','*1S(=O)(=O)(F)'],
    ['F','*1F'],
    ['Cl','*1[Cl]'],
    ['Br','*1[Br]'],
    ['I','*1I'],
    ['CFH2','*1C(F)'],
    ['CF2H','*1C(F)(F)'],
    ['CF3','*1C(F)(F)(F)'],
    ['OCFH2','*1OC(F)'],
    ['OCF2H','*1OC(F)(F)'],
    ['OCF3','*1OC(F)(F)(F)'],
    ['Fxo','*1OC=C(F)(F)'],
    ['Triflate','*1S(=O)(=O)C(F)(F)(F)'],
    ['Cyanoacetylene','*1C#CC#N'],
    ['Acrylate','*1OC(=O)C=C'],
    ['Methacrylate','*1OC(=O)C(C)=C'],
    ['TMS','*1[Si](C)(C)(C)'],
    ['TBDMS','*1[Si](C(C)(C)(C))(C)(C)'],
    ['TIPS','*1[Si](C(C)(C))(C(C)(C))(C(C)(C))'],
    ['B(OH)2','*1B(O)(O)'],
    ['Bpin','*1B0OC(C)(C)C(C)(C)O0'],
    ['BMIDA','*1B0OC(=O)CN(C)CC(=O)O0'],
    ['Pentamethyldisiloxane','*1[Si](C)(C)O[Si](C)(C)(C)'],
    ['Tetramethyldisilabutane','*1[Si](C)(C)C[Si](C)(C)(C)'],
    
    # Links - groups with two (and only 2) attachment points 
    ['Ester','*1C(=O)O*1'],
    ['Thioester','*1C(=O)S*1'],
    ['Dimethylsiloxy','*1[Si](C)(C)O*1'],
    ['Dimethylsilyl','*1[Si](C)(C)*1'],
    
    #various fluorinated things
    ['CF2O','*1C(F)(F)O*1'],
    ['CFO','*1C(F)(F)O*1'],
    ['CFHCH2','*1C(F)C*1'],
    ['CF2CH2','*1C(F)(F)C*1'],
    ['CF2CFH','*1C(F)(F)C(F)*1'],
    ['CF2CF2','*1C(F)(F)C(F)(F)*1'],
    ['CFCH(trans)','*1C(F)=C*1'],
    
    ['Methyleneoxy','*1CO*1'],
    ['Amide','*1NC(=O)*1'],
    ['Nmethylamide','*1N(C)C(=O)*1'],
    ['Imine','*1C=N*1'],
    ['Azo','*1N=N*1'],
    ['Azoxy','*1\\N=[N+](/[O-])*1'],
    
    ['C2H4','*1CC*1'],
    ['Acetylene','*1C#C*1'],
    ['C2H2','*1C=C*1'],
    ['Carbonyl','*1C(=O)*1'],
    ['Oxy','*1O*1'],
    ['Thio','*1S*1'],
    ['Seleno','*1[Se]*1'],
    ['Carbonyl','*1C(=O)*1'],
    ['Sulfoxide','*1S(=O)*1'],
    ['Sulfone','*1S(=O)(=O)*1'],
    
    #Alkyl Fragments
    ['Methyl','*1C'],
    ['Branch1','*1C(C)(C)*1'],
    ['Branch2','*1C(C)(C)C*1'],
    ['Branch3','*1C(C)*1'],
    
    #Alkyl Spacers
    ['CH2','*1C*1'],
    ['C2H4','*1CC*1'],
    ['C3H6','*1CCC*1'],
    ['C4H8','*1CCCC*1'],
    ['C5H10','*1CCCCC*1'],
    ['C6H12','*1CCCCCC*1'],
    ['C7H14','*1CCCCCCC*1'],
    ['C8H16','*1CCCCCCCC*1'],
    ['C9H18','*1CCCCCCCCC*1'],
    ['C10H20','*1CCCCCCCCCC*1'],
    ['C11H22','*1CCCCCCCCCCC*1'],
    ['C12H24','*1CCCCCCCCCCCC*1'],

    #Fluorocarbon Chains
    ['CF3','*1C(F)(F)(F)'],
    ['C2F5','*1C(F)(F)C(F)(F)(F)'],
    ['C3F7','*1C(F)(F)C(F)(F)C(F)(F)(F)'],
    ['C4F9','*1C(F)(F)C(F)(F)C(F)(F)C(F)(F)(F)'],
    ['C5F11','*1C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)(F)'],
    ['C6F13','*1C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)(F)'],
    ['C7F15','*1C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)(F)'],
    ['C8F17','*1C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)(F)'],
    ['C9F19','*1C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)(F)'],
    ['C10F21','*1C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)(F)'],
    ['C11F23','*1C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)(F)'],
    ['C12F25','*1C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)(F)'],
    ]

    # process the fragments into Groups
    GroupFrags=[]
    for x in range(len(Groups)):
        GroupFrags.append (Group(str(Groups[x][0]),str(Groups[x][1])))

    RingFrags=[]
    for x in range(len(Rings)):
        RingFrags.append (Group(str(Rings[x][0]),str(Rings[x][1]),all_attachment=True))
    
    # Process the combined groups into a Grammar
    Grammar = GroupGrammar(RingFrags+GroupFrags)
 
    return(Grammar,Rings)

def GrammarFromFrag(Frag):

    Grammar = GroupGrammar(Frag)
    
    return(Grammar)

def FragmentMol(Frag,MolObject):
    Grammar = GroupGrammar(Frag)
    extracted = Grammar.extract_groups(MolObject)
    
    for group, _, _ in extracted:
        print('Group Name:', group.name)
        display(group.mol)
        
    print(Grammar.encoder(MolObject, extracted))
    return(extracted)
    
def newfrag(grammar_fragment,Additions,Branches):
    '''
    See https://github.com/aspuru-guzik-group/group-selfies/blob/main/tutorial/tutorial.ipynb
    
    This has been changed slightly so that when we call for a new fragment (in RandomGSel)
    we also ask for a random number of Additions and Branches.
    
    Additions controls the number of fragments in the returned gselfies, while Branches
    controls the number of pop tokens.
    
    Inputs:
    grammar_fragment    - the Grammar we will use to look up groups/fragments
    Additions           - the number of groups to add
    Branches            - the number of branches to add (via [pop])
    
    returns
    new_gselfies        - a string which is a new group-selfies fragment.
    '''
    
    _,Rings = DefinedFragments() # loads rings
    
    for group_name in grammar_fragment.vocab:
        group = grammar_fragment.vocab[group_name]
        group.overload_idx = random.randint(0, 100)

    group_names = list(grammar_fragment.vocab.keys())
    

    random_groups = [random.choice(group_names) for _ in range(Additions)] + ['[pop]' for _ in range(Branches)]
    random.shuffle(random_groups)
    
    # combine them into a new group selfies string

    new_gselfies = ''
    for g in random_groups:
        if g == '[pop]':
            new_gselfies += '[pop]'
            continue
        if g == '[Branch]':
            new_gselfies += '[Branch]'
            continue
            
        start = random.randint(0, len(grammar_fragment.vocab[g].attachment_points)-1)
        #new_gselfies +=  ('[Branch]' * abs(Hit-1)) + ('[Ring1]' * Hit) + f"[:{start}{g}]" + ('[pop]' * abs(Hit-1)) + ('[Branch]' * Hit) # interesting

        new_gselfies +=  f"[:{start}{g}]" + ('[Branch]' * (str(g) in itertools.chain(*Rings)))# interesting
        
    return(new_gselfies)

def CleanFlags(string):
    '''
    Takes group-selfies as a list of strings and removes cases where we have two
    ring flags. Could probably stop that happening ratehr than have to fix it.
    
    inputs:
    string - a string which is the joined group-selfies, basically.
    
    returns
    new_string - much the same, but any adjacent ring flags are removed.
    '''
   
    # Define a regular expression pattern that matches consecutive [Ringx] flags.
    # can also find [Ring(x)] when adjacent to [Branch] (which it shouldn't be...)
    pattern = r"\[Ring(\d+)\]\[Ring\1\]|\[Branch\]\[Ring(\d+)\]|\[Ring(\d+)\]\[Branch\]"

    # Replace any matches of the pattern with an empty string.
    new_string = re.sub(pattern, "", string)

    return(new_string)

def RandomGSel(Mol,Grammar,Silent=True,Continues=4,AddChances=4,BranchChances=2,InsertChance=40,ReplaceChance=40,Cleanup=False):

    '''
    Silent - if True, don't print stuff to the workspace. 
    Continues - this many chances to continue (out of continues+1) making changes;
                a bigger number leads to more molecular changes.
                
    inputs:
    mol             - the mol object we are going to be modifying
    grammar         - the grammar (framgent list) we'll be inserting/replacing/deleting
    silent          - keep it quiet - don't dump loads of stuff to the user
    Continues       - how many times (on average) we should make a change.
    AddChancse      - when inserting/replacing the max number of additions (i.e fragments) to add
    BrancChances    - the max number of branches during insert/replace activities
    InsertChance    - the chance (in %) that we'll do an insertion
    ReplaceChance   - the chance (in %) that we'll do a replacement
    
    Other stuff
    DeleteChance    - calcualted on the fly as 100 - (InsertChance + ReplaceChance)
    
    returns:
    NewGrammar - The modified group selfies string.
    '''
    
    Action = [1] * InsertChance + [2] * ReplaceChance + [3]*(round(100-(InsertChance+ReplaceChance))) #1 = insert, 2 = replace, 3 = deletion
    GoAgain = 1

    Seq = [] # just a list we'll use for some logging later on.
        
    while GoAgain == 1:     
        # while GoAgain = 1 we keep making changes
        # a crude way to enforce multiple random alterations.
        # keeping these lines in the while loop keeps it dynamic
        Grammar,_ = DefinedFragments()
                     
        SplitGrammar = re.findall(r'\[.*?\]',Grammar.encoder(Mol,Grammar.extract_groups(Mol)))  
        
        Random_Action = np.random.choice(Action,1)[0] # do a random action
        
        Seq.append(str(Random_Action)) # record the random action so we can remind ourselves later
        
        Idx = 1+np.random.randint(len(SplitGrammar)-2) # place to act; we don't want to do stuff at Idx=0 because it'll break the moleucle quite often.
        
        while re.findall(r'\[Ring.*?\]|\[Branch\]|\[pop\]',str(SplitGrammar[Idx])):
            Idx = 1+np.random.randint(len(SplitGrammar)-2) # place to act; we don't want to do stuff at Idx=0 because it'll break the moleucle quite often.

        if AddChances <=1:
            Additions = AddChances
            
        if AddChances >=2:
            Additions = random.choice(list(range(1,AddChances)))
        
        if BranchChances <=1:
            Branches = BranchChances
            
        if BranchChances >=2:
            Branches = random.choice(list(range(1,BranchChances)))
        
        NewFrag = newfrag(Grammar,Additions,Branches) #group we will add
        
        if Random_Action == 1: #insert something
            if not Silent:
                print('Insertion of ' + str(NewFrag) + ' at Idx = ' + str(Idx))
                print('Current Fragments:\n'+
                    str(Idx-1) + '  ' + SplitGrammar[Idx-1] + '\n' + 
                    '** INSERTING HERE **\n' + 
                    str(Idx) + '  ' + SplitGrammar[Idx] +  '\n' +
                    str(Idx+1) + '  ' + SplitGrammar[Idx+1] + '\n')
            
            #if SplitGrammar[Idx] == '[pop]':
            #    NewFrag = '[pop]' + NewFrag
                
            SplitGrammar.insert(Idx,NewFrag)

        if Random_Action == 2: #replace something
            if not Silent:
                print('replacement of ' + SplitGrammar[Idx] + ' at Idx = ' + str(Idx) + ' with ' + str(NewFrag))
                print('Current Fragments:\n'+
                    str(Idx-1) + '  ' + SplitGrammar[Idx-1] + '\n' + 
                    str(Idx) + '  ' + SplitGrammar[Idx] + ' <-- ** REPLACE **\n' + 
                    str(Idx+1) + '  ' + SplitGrammar[Idx+1] + '\n')
                    
            if SplitGrammar[Idx] == '[pop]': # preserve [pop] tags
                NewFrag =  NewFrag + '[pop]'
            
            if SplitGrammar[Idx] == '[Branch]': # preserve [Branch] tags
                NewFrag =  NewFrag + '[Branch]'   
                
            SplitGrammar[Idx] = NewFrag

        if Random_Action == 3: #delete something
            if not Silent:
                print('Deletion of ' + SplitGrammar[Idx] + ' at Idx = ' + str(Idx))
                print('Current Fragments:\n'+
                    str(Idx-1) + '  ' + SplitGrammar[Idx-1] + '\n' + 
                    str(Idx) + '  ' + SplitGrammar[Idx] + ' <-- ** DELETE **\n' + 
                    str(Idx+1) + '  ' + SplitGrammar[Idx+1] + '\n')
                    
            #only delete if its not a ring-bond. this just messes things up badly.    
            if not re.findall(r'\[Ring.*?\]',str(SplitGrammar[Idx])):
                SplitGrammar[Idx] = ''
            
        #convert it into canonical gselfies
        
        if Cleanup:
            #print('Precanonical grammar:\n' + str(''.join(SplitGrammar)))
            SplitGrammar = CleanFlags(str(''.join(SplitGrammar)))
           
            #print('Cleaned Ring grammar:\n' + str(''.join(SplitGrammar)))
            temp_smi = Chem.MolToSmiles(Grammar.decoder(str(''.join(SplitGrammar)))) # get smi from encoded grammar
            temp_mol = Chem.MolFromSmiles(temp_smi)                                  # get that mol object

            SplitGrammar = Grammar.encoder(temp_mol,Grammar.extract_groups(temp_mol))

        if not Cleanup:
            #stick it together and see what happens
            SplitGrammar = str(''.join(SplitGrammar))
        
        if not Silent:
            #if not silent, show the change at each iteration both in terms of gselfies and the drawn molecule
            print('\nCurrent Grammar:\n' + SplitGrammar)
            display(Grammar.decoder(str(''.join(SplitGrammar)))) 

        Mol = Chem.MolFromSmiles(Chem.MolToSmiles(Grammar.decoder(SplitGrammar))) # This keeps the mol object clean and canonical
          
        GoAgain = np.random.choice([1]*Continues+[0])
        
        if SplitGrammar.count('[') == 1: #stop if you've deleted all but one fragment...
            print('Stopping beore we run out of fragments to delete...')
            GoAgain = 0
        
    
    #if not Silent:    
    #    print('\n Final Grammar ' + SplitGrammar)
    #    print('\nI performed the following opperations in this order: ')
    #    print(str(Seq).replace('1','Insertion /').replace('2','Replacement /').replace('3','Deletion /'))
 
    NewGrammar = Grammar.decoder(SplitGrammar)
    
    if not Silent:
        print('Final generated molecule above')
    return(NewGrammar)    