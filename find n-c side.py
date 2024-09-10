import pandas as pd


file_path = 'path_to_your_file.xlsx'
df = pd.read_excel(file_path)


residues_df = df[['First_Residue', 'Last_Residue']].copy()


residues_df.loc[:, 'First_Residue'] = residues_df['First_Residue'].str.extract(r"\('(\w)', '(\w)'\)").apply(lambda x: ''.join(x), axis=1)
residues_df.loc[:, 'Last_Residue'] = residues_df['Last_Residue'].str.extract(r"\('(\w)', '(\w)'\)").apply(lambda x: ''.join(x), axis=1)

# Delete only the rows that contain 'X' in the Last_Residue column
filtered_df = residues_df[~residues_df['Last_Residue'].str.contains('X')]


unique_df = filtered_df.drop_duplicates()


output_file_path = 'n-c side.xlsx'  # 保存为 'n-c side.xlsx'
unique_df.to_excel(output_file_path, index=False)