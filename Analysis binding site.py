import os
import pandas as pd
from Bio import PDB

#Identify the binding site between the two chains
def identify_binding_sites(chain1, chain2, distance_threshold=5.0):
    binding_residues = []
    for residue1 in chain1:
        for residue2 in chain2:
            for atom1 in residue1:
                for atom2 in residue2:
                    if atom1 - atom2 < distance_threshold:
                        binding_residues.append((residue1, residue2))
                        break
                else:
                    continue
                break
            else:
                continue
            break

    return binding_residues


#Extract sequence information of binding sites
def extract_sequences(binding_residues):
    sequences = []
    for residue1, residue2 in binding_residues:
        sequences.append({
            "Chain1_Residue": residue1.get_resname(),
            "Chain1_Residue_ID": residue1.get_id()[1],
            "Chain2_Residue": residue2.get_resname(),
            "Chain2_Residue_ID": residue2.get_id()[1]
        })
    return pd.DataFrame(sequences)


#Process all PDB files in a directory, extract binding site information, and save to Excel files
def process_pdb_files(input_dir, output_file):
    all_data = []
    pdb_parser = PDB.PDBParser(QUIET=True)
    pdb_files = [f for f in os.listdir(input_dir) if f.endswith(".pdb")]
    total_files = len(pdb_files)

    for index, filename in enumerate(pdb_files):
        file_path = os.path.join(input_dir, filename)
        print(f"Processing files: {file_path} ({index + 1}/{total_files})")
        structure = pdb_parser.get_structure(filename, file_path)
        model = structure[0]

        chains = list(model.get_chains())
        num_chains = len(chains)

        for i in range(num_chains):
            for j in range(i + 1, num_chains):
                chain1_id = chains[i].id
                chain2_id = chains[j].id
                binding_residues = identify_binding_sites(chains[i], chains[j])

                if binding_residues:
                    sequence_df = extract_sequences(binding_residues)
                    sequence_df["PDB_File"] = filename
                    sequence_df["Chain1"] = chain1_id
                    sequence_df["Chain2"] = chain2_id
                    all_data.append(sequence_df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_excel(output_file, index=False)
        print(f"Successfully saved binding site information to Excel file: {output_file}")
    else:
        print("No binding site was found.")



input_dir = "fasta_files"
output_file = "binding_sites.xlsx"
process_pdb_files(input_dir, output_file)