import os
import torch
import pandas as pd
torch.set_printoptions(threshold=1000)
torch.set_printoptions(edgeitems=5)
# List of parent directories

parent_dirs = ["hx1xuegl",] # List of Teacher IDs you wish to ensemble

base_path = r"D:/Sean/github/DCASE2024_Task1/predictions/" # Change path as needed

# Initialize a list to store logits tensors
logits_list = []

# Iterate over the directories and read the logits
for dir_name in parent_dirs:
    csv_file = os.path.join(base_path, dir_name, "output.csv")
    df = pd.read_csv(csv_file, delimiter='\t')
    logits = df.iloc[:, 2:].values  # Assuming the logits start from the third column
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    logits_list.append(logits_tensor)

# Compute the average of logits tensors
average_logits = torch.mean(torch.stack(logits_list), dim=0)

# Define the path to save the averaged logits as a .pt file in a new "ensemble" folder
ensemble_dir = os.path.join(base_path, "ensemble")
os.makedirs(ensemble_dir, exist_ok=True)
###### REMEMBER TO CHANGE PT NAME#############

pt_file = os.path.join(ensemble_dir, "sub5_single_PaSST_demo.pt") # change name as needed. Be careful not to overwrite your logits
###### REMEMBER TO CHANGE PT NAME#############
# Save the averaged logits tensor to the .pt file
torch.save(average_logits, pt_file)

print("Logits saved to", pt_file)

# Load the saved averaged logits from the .pt file
loaded_logits = torch.load(pt_file)

# Print the loaded averaged logits tensor
print("Loaded logits:")
print(loaded_logits)
