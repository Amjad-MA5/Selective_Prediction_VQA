from datasets import load_dataset, load_from_disk
import os

folder_path = '/teamspace/studios/this_studio/Selective_Prediction_PathVQA/dataset/data/VQAv2.hf'
dataset_script_path = '/teamspace/studios/this_studio/Selective_Prediction_PathVQA/dataset/VQAv2.py'

def get_VQAv2_dataset():
    if os.path.isdir(folder_path):
        print(f"Loading from local path {folder_path}.")
        dataset = load_from_disk(folder_path)
    else:
        print("Downloading from source")
        from dataset.download_VQAv2_dataset import download_dataset
        dataset = download_dataset(folder_path)
        
    return dataset

if __name__ == "__main__":
    dataset = get_VQAv2_dataset()