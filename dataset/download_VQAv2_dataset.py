from datasets import load_dataset

import sys
sys.path.insert(0, "/teamspace/studios/this_studio/Selective_Prediction_PathVQA/dataset/download_VQAv2_dataset.py")

def download_dataset(folder_path):
    dataset = load_dataset('VQAv2.py', name='v2')
    dataset.save_to_disk(folder_path)
    return dataset


if __name__ == "__main__":
    download_dataset("./data/VQAv2.hf")
