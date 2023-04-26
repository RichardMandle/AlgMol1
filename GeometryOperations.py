import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

'''
Contains functions for taking text representations of molecules (smiles)
and generating 3D coordinates, performing conformer searching in RDKit, and
setting up calculation input files for external programs.

functions:
Minimum_Energy_Conformer - does a conformer search using RDkit and the MMFF method
WriteGauss               - Creates a Gaussian input file for a given geometry
WriteMopac               - Creates a Mopac input file for a given geometry
MakeSGEJob               - Makes a job submission script for the SGE system as configured on ARC3/4 at UoL
'''

def Minimum_Energy_Conformer(smiles,num_of_conformer=500,max_iter=1000,min_energy_MMFF=10000):
    '''
    Takes an input smiles string and uses rdkit to perform a conformer search with the MMFF method.
    
    inputs:
    smiles - the smiles string to work with
    num_of_conformer - the number of conformers to use
    max_iter - maximum iterations
    min_energy_MMFF - the energy threshold for the conformer search
    
    returns:
    mol_min - the minimum energy conformation as an rdkit Mol object
    '''
    mol = Chem.MolFromSmiles(smiles)
    mol_h_MMFF = Chem.AddHs(mol)
    min_energy_index_MMFF=0

    # Generate conformers (stored in side the mol object)
    cids = AllChem.EmbedMultipleConfs(mol_h_MMFF, numConfs=num_of_conformer,params=AllChem.ETKDG())

    results_MMFF = AllChem.MMFFOptimizeMoleculeConfs(mol_h_MMFF,maxIters=max_iter)

    # Search for the min energy conformer from results(tuple(is_converged,energy))

    print("\nSearching conformers by MMFF ")   
    for index, result in enumerate(results_MMFF):
        if(min_energy_MMFF>result[1]):       
            min_energy_MMFF=result[1]
            min_energy_index_MMFF=index
            print(min_energy_index_MMFF,":",min_energy_MMFF)

    mol_min=(Chem.Mol(mol_h_MMFF,False,min_energy_index_MMFF))
    Draw.MolToImage(mol_min)

    return(mol_min)

def WriteGauss(smiles,
               file_name='smiles',
               functional='AM1',
               basis='',
               options='',
               num_of_conformer=500,
               max_iter=1000,
               min_energy_MMFF=10000,
               mem=2,
               nproc=1,):
    """
    This function prepares a Gaussian input file from the given smiles using rdkit to prepare initial coordintes by
    1. Converting SMILES (smiles) to molecule
    2. Preparing 3D coordinates
    3. Writing out Gaussian input file
    
    On the offchance the user only supplies a smiles string, we request a semi-empirical calculation (AM1)
    'options' allows us to insert other keywords, read the g09 manual.
    
    Useful function to call in a loop to write an array of smiles strings to .gjf files
    
    inputs:
    smiles     - the smiles string to work with
    file_name  - the filename for the .gjf file (and .log file output)
    functional - the method to use (e.g. AM1, B3LYP, MP2...)
    basis      - the basis set to use (e.g. '','6-31G','aug-ccpVTZ...')
    options    - any additional calculation options, e.g. FREQ, POLAR, iop(7/33=1)
    
    num_of_conformer - 500 - the number of conformers to use in the (initial) MMFF confo. search
    max_iter         - maximum iteractions for conformer search
    min_energy_MMFF  - minimum energy for conformer search
    
    mem   - ammount of memory (in GB) to use for Gaussian calculations
    nproc - number of processors to use for Gaussian calculations
    
    returns:
    the filename of the calculation (but why?)
    saves a .gjf file in the current path
    
    """
    
    file_name = file_name + '.gjf'         # Write out Gaussian input file
    
    mol = Minimum_Energy_Conformer(smiles,num_of_conformer,max_iter,min_energy_MMFF)
    
    with open(file_name, 'w') as f:
        #f.write('%chk=' + smiles + '.chk\n')
        f.write('%nprocshared=' + str(nproc) + '\n')
        f.write('%mem=' + str(mem) + 'GB\n')
        f.write('#p opt freq polar' + functional + ' ' + basis + ' ' + options + '\n\n')
        f.write(smiles + '\n\n')
        f.write('0 1\n')
        
        for i, atom in enumerate(mol.GetAtoms()):
            positions = mol.GetConformer().GetAtomPosition(i)
            f.write(atom.GetSymbol() + '\t' + str(positions.x) + '\t' + str(positions.y) + '\t' + str(positions.z))
            f.write('\n')
    
    return file_name

def WriteMopac(smiles,
               file_name='smiles',
               kernel='AM1',
               options='PRECISE EF',
               time='48H',
               num_of_conformer=500,
               max_iter=1000,
               min_energy_MMFF=10000):

    """
    This function prepares a Mopac input file from the given smiles using rdkit to prepare initial coordintes by
    1. Converting SMILES (smiles) to molecule
    2. Preparing 3D coordinates
    3. Writing out Mopac 'dat' input file
    
    Useful function to call in a loop to write an array of smiles strings to .dat files
    
    inputs:
    smiles           - the smiles string to work with
    file_name        - the filename for the .gjf file (and .log file output)
    kernel           - the method to use (e.g. AM1, PM3, PM6, PM7, etc)
    options          - any additional calculation options, defaults to "PRECISE" and "EF"
    time             - the time alloted for the job, default of 48H so it aligns with the max on ARC3/4
    num_of_conformer - 500 - the number of 
    to use in the (initial) MMFF confo. search
    max_iter         - maximum iteractions for conformer search
    min_energy_MMFF  - minimum energy for conformer search
    
    returns:
    the filename of the calculation (but why?)
    saves a .dat file in the current path
    
    """
    
    file_name = file_name + '.dat'         # Write out Gaussian input file
    
    mol = Minimum_Energy_Conformer(smiles,num_of_conformer,max_iter,min_energy_MMFF)
    
    with open(file_name, 'w') as f:
        f.write(kernel + ' ' + options + 'T='+time + '\n')
        f.write(smiles + '\n\n')
        f.write('0 1\n')
        
        for i, atom in enumerate(mol.GetAtoms()):
            positions = mol.GetConformer().GetAtomPosition(i)
            f.write(atom.GetSymbol() + '\t' + str(positions.x) + '\t' + str(positions.y) + '\t' + str(positions.z))
            f.write('\n')
    
    return file_name

def MakeSGEJob(filename='noname', vmem=2,nproc=1,startjob=0,endjob=0):
    '''
    Makes a simple .sh file for submission to the HPC queue at UoL via SGE
    '''
    
    with open(filename + '.sh', 'w') as f:
        f.write('#$ -cwd \n') 
        f.write('#$ -V\n')  
        f.write('#$ -l h_rt=48:00:00\n') 
        f.write('#$ -l h_vmem=' + str(vmem) + 'G\n') 
        f.write('#$ -pe smp ' + str(nproc) + '\n') 
        f.write('#$ -m be\n')
        #f.write('#$ -M x.xxxxxx@leeds.ac.uk\n') # put your email here
        f.write('#$ -l disk=5G\n')
        f.write('#$ -t ' + str(startjob) + '-' + str(endjob) + '\n') # create a task array

        f.write('module add gaussian\n')
        f.write('export GAUSS_SCRDIR=$TMPDIR\n')
        f.write('g09 ' + filename +'_[$SGE_TASK_ID].gjf\n')
