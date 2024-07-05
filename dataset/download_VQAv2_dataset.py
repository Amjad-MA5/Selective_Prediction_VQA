from datasets import load_dataset


def download_dataset():
    dataset = load_dataset('VQAv2.py', name='v2')
    dataset.save_to_disk("./data/VQAv2.hf")
    return dataset


if __name__ == "__main__":
    download_dataset()
