import os
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm


excel_file = '/Users/zephyr/Documents/PycharmProjects/igem/pdl1/binding_sites_single.xlsx'
df = pd.read_excel(excel_file)
pdb_folder = '/Users/zephyr/Documents/PycharmProjects/igem/pdl1/fasta_files'


parser = PDBParser(QUIET=True)


aa_three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


# Define a function to get amino acid information from the PDB file
def get_residue_info(pdb_id, chain_id, residue_id, parser, pdb_folder):
    pdb_path = os.path.join(pdb_folder, pdb_id)
    structure = parser.get_structure(pdb_id, pdb_path)
    model = structure[0]
    chain = model[chain_id]
    residues = list(chain.get_residues())

    first_res = aa_three_to_one.get(residues[0].get_resname(), 'X')
    second_res = aa_three_to_one.get(residues[1].get_resname(), 'X') if len(residues) > 1 else 'X'
    last_res = aa_three_to_one.get(residues[-1].get_resname(), 'X')

    # Find the index of the target residue
    target_index = next((i for i, res in enumerate(residues) if res.id[1] == residue_id), None)

    if target_index is not None:
        start = max(0, target_index - 4)
        end = min(len(residues), target_index + 5)
        neighbor_residues = [aa_three_to_one.get(res.get_resname(), 'X') for res in residues[start:end]]
    else:
        neighbor_residues = []

    return first_res, second_res, last_res, neighbor_residues



first_residues = []
second_residues = []
last_residues = []
neighbor_residues_list = []

# Iterate over each row in a DataFrame, extract information and add it to a new column
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    pdb_file = row['PDB_File']
    chain1 = row['Chain1']
    chain2 = row['Chain2']
    residue_id1 = row['Chain1_Residue_ID']
    residue_id2 = row['Chain2_Residue_ID']

    first_res1, second_res1, last_res1, neighbors1 = get_residue_info(pdb_file, chain1, residue_id1, parser, pdb_folder)
    first_res2, second_res2, last_res2, neighbors2 = get_residue_info(pdb_file, chain2, residue_id2, parser, pdb_folder)

    first_residues.append((first_res1, first_res2))
    second_residues.append((second_res1, second_res2))
    last_residues.append((last_res1, last_res2))
    neighbor_residues_list.append((neighbors1, neighbors2))

df['First_Residue'] = first_residues
df['Second_Residue'] = second_residues
df['Last_Residue'] = last_residues
df['Neighbor_Residues'] = neighbor_residues_list


output_file = 'updated_binding_sites.xlsx'
df.to_excel(output_file, index=False)