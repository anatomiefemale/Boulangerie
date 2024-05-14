# Part 1: Obtain Data
#______________________________________________________________________________
'''
This setup will allow you to effectively access chemical data from PubChem.
'''

### Step 1: Install `pubchempy`

#pip install pubchempy

### Step 2: Using `pubchempy` to Retrieve SMILES
'''
Here's a basic example of how you can retrieve the SMILES 
string for a specific compound by its name, 
CAS number, or CID (Compound ID).
'''
#### Example 1: Retrieve SMILES by Compound Name

import pubchempy as pcp

def get_smiles_by_name(compound_name):
    try:
        compound = pcp.get_compounds(compound_name, 'name')[0]  # Get the first matching compound
        return compound.isomeric_smiles  # Return the isomeric SMILES string
    except IndexError:
        return "Compound not found"

# Example usage
compound_name = 'Aspirin'
smiles_string = get_smiles_by_name(compound_name)
print(f'SMILES for {compound_name}: {smiles_string}')
# SMILES for Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O

#### Example 2: Retrieve SMILES by CID

def get_smiles_by_cid(cid):
    try:
        compound = pcp.Compound.from_cid(cid)
        return compound.isomeric_smiles
    except Exception as e:
        return str(e)

# Example usage
cid = 2244  # CID for Aspirin
smiles_string = get_smiles_by_cid(cid)
print(f'SMILES for CID {cid}: {smiles_string}')
# SMILES for CID 2244: CC(=O)OC1=CC=CC=C1C(=O)O

### Step 3: Handling Multiple Compounds and Advanced Queries
'''
Note: PubChemPy allows for more complex queries and handling multiple results.
'''
def search_smiles(query):
    compounds = pcp.get_compounds(query, 'name')
    smiles_list = [comp.isomeric_smiles for comp in compounds if comp.isomeric_smiles is not None]
    return smiles_list

# Example usage
query = 'benzene'
smiles_results = search_smiles(query)
print(f'SMILES for {query}: {smiles_results}')
# SMILES for benzene: ['C1=CC=CC=C1']

# Part 2: Train LLM
#______________________________________________________________________________

import torch
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem

# Create a custom dataset for SMILES strings
class SmilesDataset(Dataset):
    def __init__(self, smiles, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = [tokenizer.encode(smile, max_length=max_length, truncation=True) for smile in smiles]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

# Sample data (usually you'll have much more data)
smiles_data = ["CCO", "O=C(O)c1ccccc1C(=O)O", "CCCC"]

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Prepare dataset and dataloader
dataset = SmilesDataset(smiles_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Example training loop (simplified)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(2):  # This would be much higher in a real scenario
    for batch in dataloader:
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Training loss: {loss.item()}")

# Generating a new molecule (basic example) aka a SMILES string
model.eval()
sampled_smiles = "CC(C)"
input_ids = tokenizer.encode(sampled_smiles, return_tensors="pt")
with torch.no_grad():
    predictions = model(input_ids)[0]
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_token = tokenizer.decode([predicted_index])
    new_smiles = sampled_smiles + predicted_token

# Validate new SMILES
new_smiles = new_smiles.strip(",.!? ")  # Remove potentially problematic characters
new_mol = Chem.MolFromSmiles(new_smiles)
if new_mol:
    print(f"Generated valid SMILES: {new_smiles}")
else:
    print("Generated invalid SMILES")
# Generated valid SMILES: CC(C)
"""
When you receive an output like "Generated valid SMILES: CC(C)", 
it indicates that the SMILES string "CC(C)" has been successfully 
recognized and validated as a correct representation of a chemical 
structure using the RDKit library. 
"""
