# download_VQAv2_dataset.py

from datasets import load_dataset

def download_dataset(folder_path):
    dataset = load_dataset('/teamspace/studios/this_studio/Selective_Prediction_PathVQA/dataset/VQAv2.py', name='v2')
    dataset.save_to_disk(folder_path)
    return dataset

if __name__ == "__main__":
    download_dataset("/teamspace/studios/this_studio/Selective_Prediction_PathVQA/dataset/data/VQAv2.hf")