import pandas as pd


three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y'
}

input_file = "binding_sites.xlsx"
output_file = "binding_sites_single.xlsx"
sheet_name = "Sheet1"


df = pd.read_excel(input_file, sheet_name=sheet_name)


def convert_to_single_letter(sequence):
    if isinstance(sequence, str):
        return ''.join(three_to_one.get(residue, 'X') for residue in sequence.split())
    else:
        return 'X'  # Or return an empty string '' if you want to keep it as a null value

# Transform the relevant columns to ensure that NaN and non-string data are handled
df['Chain1_Residue'] = df['Chain1_Residue'].apply(lambda x: convert_to_single_letter(x) if pd.notna(x) else '')
df['Chain2_Residue'] = df['Chain2_Residue'].apply(lambda x: convert_to_single_letter(x) if pd.notna(x) else '')


df.to_excel(output_file, index=False)