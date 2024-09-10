import requests
from Bio import PDB
import os
from tqdm import tqdm


def download_pdb(pdb_id, output_file):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, 'w') as file:
            file.write(response.text)
        return True
    else:
        print(f"Download {pdb_id} failed，Status Code：{response.status_code}")
        return False


def pdb_to_fasta(pdb_file, fasta_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    with open(fasta_file, 'w') as file:
        ppb = PDB.PPBuilder()
        for model in structure:
            for chain in model:
                sequence = ""
                for residue in chain:
                    if PDB.is_aa(residue, standard=True):
                        # Note: This might be incorrect if there are multiple peptide chains in one PDB file
                        sequence += ppb.build_peptides(chain, aa_only=True)[0].get_sequence()

                if sequence:
                    file.write(f">{chain.get_id()}\n")
                    file.write(f"{sequence}\n")


def process_pdb_files(input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r') as file:
        pdb_ids = [line.strip() for line in file.readlines() if line.strip()]

    for pdb_id in tqdm(pdb_ids, desc="Processing PDB Files"):
        pdb_file = os.path.join(output_dir, f"{pdb_id}.pdb")
        fasta_file = os.path.join(output_dir, f"{pdb_id}.fasta")

        # Download the PDB file if it doesn't already exist
        if not os.path.exists(pdb_file):
            success = download_pdb(pdb_id, pdb_file)
            if not success:
                continue  # Skip processing this PDB file if download failed

        # Convert PDB to FASTA
        pdb_to_fasta(pdb_file, fasta_file)



input_file = "pdb_ids.txt"
output_dir = "fasta_files"
process_pdb_files(input_file, output_dir)