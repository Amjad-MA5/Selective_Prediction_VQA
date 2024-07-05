from datasets import load_dataset
import os

folder_path = './data/VQAv2.hf'


# Ensure the dataset script file is in your PYTHONPATH or specify the path to the script
def get_VQAv2_dataset():
    if os.path.isdir(folder_path):
        print(f"Loading from local path {folder_path}.")
        dataset = load_dataset('VQAv2.py', data_dir=folder_path, name='v2')
    else:
        print(f"Downloading from source")    
        dataset = load_dataset('VQAv2.py', name='v2')
    
    return dataset



if __name__ == "__main__":
    dataset = get_VQAv2_dataset()