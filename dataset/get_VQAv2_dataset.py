from datasets import load_dataset, load_from_disk
import os

home_dir = "/teamspace/studios/this_studio"
folder_path =  home_dir + "/Selective_Prediction_PathVQA/dataset/data/VQAv2.hf"
script_path = home_dir + "/Selective_Prediction_PathVQA/dataset/VQAv2.py"

# Ensure the dataset script file is in your PYTHONPATH or specify the path to the script

def get_VQAv2_dataset():
    print(folder_path)
    if os.path.isdir(folder_path):
        print(f"Loading from local path {folder_path}.")
        dataset = load_from_disk(folder_path)
    else:
        print(f"Downloading from source")
        try:
            from download_VQAv2_dataset import download_dataset
        except:
            pass
        dataset = download_dataset(folder_path)
        
    return dataset



if __name__ == "__main__":
    dataset = get_VQAv2_dataset()