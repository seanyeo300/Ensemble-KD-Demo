import os
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

torch.set_printoptions(threshold=1000)
torch.set_printoptions(edgeitems=5)
# List of parent directories
# DCASE ensemble (2 SIT 2 PaSST 2 BCBL, same augs)    
parent_dirs = ["fskag87u","bxgn5l84","jiw5bohu", "erxj7yo6", "huyzahj3", "ttpwu2wq"]    # Change to your own directories

base_path = r"D:\Sean\github\cpjku_dcase23_NTU\predictions" # Change path as needed
test_csv_path = r"D:\Sean\github\cpjku_dcase23_NTU\split_setup\test.csv"

# Initialize a list to store logits tensors
logits_list = []
filename_list = None  # Will hold the filenames in order

# Iterate over the directories and read the logits
for dir_name in parent_dirs:
    csv_file = os.path.join(base_path, dir_name, "output.csv")
    df = pd.read_csv(csv_file, delimiter='\t')
    df['filename'] = df['filename'].str.strip().str.lower()
    if filename_list is None:
        filename_list = df['filename'].tolist()
    logits = df.iloc[:, 2:].values  # Assuming the logits start from the third column
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    logits_list.append(logits_tensor)

# Compute the average of logits tensors
average_logits = torch.mean(torch.stack(logits_list), dim=0)

# Define the path to save the averaged logits as a .pt file in a new "ensemble" folder
ensemble_dir = os.path.join(base_path, "ensemble")
os.makedirs(ensemble_dir, exist_ok=True)
###### REMEMBER TO CHANGE PT NAME#############

pt_file = os.path.join(ensemble_dir, "dummy.pt") # change name as needed. Be careful not to overwrite your logits
###### REMEMBER TO CHANGE PT NAME#############
# Save the averaged logits tensor to the .pt file
torch.save(average_logits, pt_file)

print("Logits saved to", pt_file)

# Load the saved averaged logits from the .pt file
loaded_logits = torch.load(pt_file)

# Load and normalise test.csv
test_df = pd.read_csv(test_csv_path, delimiter='\t')
test_df['filename'] = test_df['filename'].str.strip().str[6:].str.lower()  # Remove 'audio/' prefix and normalise

# Create mapping from filename to row index
filename_to_index = {fname: idx for idx, fname in enumerate(filename_list)}

# Get indices in same order as test_df
valid_indices = [filename_to_index[fname] for fname in test_df['filename'] if fname in filename_to_index]

# Filter logits
filtered_logits = loaded_logits[valid_indices]

# Predict and compute accuracy
predicted_labels = torch.argmax(filtered_logits, dim=1).numpy()
true_labels_str = test_df['scene_label'].values
label_encoder = LabelEncoder()
label_encoder.fit(true_labels_str)
true_labels = label_encoder.transform(true_labels_str)

accuracy = accuracy_score(true_labels, predicted_labels)

# Output results
print("Filtered logits:")
print(filtered_logits)
print(f"Accuracy: {accuracy * 100:.3f}%")


# predicted_labels = torch.argmax(loaded_logits, dim=1).numpy()

# # Load and normalise test.csv
# test_df = pd.read_csv(test_csv_path, delimiter='\t')
# test_df['filename'] = test_df['filename'].str.strip().str[6:]  # Remove 'audio/' prefix
# target_filenames = set(test_df['filename'].str.lower())
# true_labels_str = df['scene_label'].values
# label_encoder = LabelEncoder()
# label_encoder.fit(true_labels_str)  # fit on ground truth
# true_labels = label_encoder.transform(true_labels_str)
# # Compute accuracy
# accuracy = accuracy_score(true_labels, predicted_labels)

# # Print the loaded averaged logits tensor
# print("Loaded logits:")
# print(loaded_logits)
# print(f"Accuracy: {accuracy * 100:.3f}%")
