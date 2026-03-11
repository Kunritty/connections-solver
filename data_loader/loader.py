from datasets import load_dataset

DATASET_ID = "tm21cy/NYT-Connections"


def load_huggingface_dataset():
    return load_dataset(DATASET_ID)


def load_connections_from_hf(split: str = "train"):
    """Load NYT Connections from Hugging Face and return a single split.

    The split has columns: date, contest, words (16),
    answers (4 groups with description + words).
    """
    ds = load_dataset(DATASET_ID)
    if split not in ds:
        split = list(ds.keys())[0]
    return ds[split]


def load_csv_dataset():
    data_files = {
        "train": "data/train_split_data.csv",
        "test": "data/test_split_data.csv"
    }
    
    dataset = load_dataset("csv", data_files=data_files)
    
    return dataset

if __name__ == "__main__":
    
    data = load_huggingface_dataset()
    print("Dataset type:", type(data))
    print("Dataset structure:", data)
    
    print("\n--- First 10 Elements ---")
    # Hugging Face datasets typically contain splits like 'train'
    split = 'train' if 'train' in data else list(data.keys())[0]
    
    for i in range(min(10, len(data[split]))):
        print(f"\nElement {i + 1}:")
        print(data[split][i])
        
    data2 = load_csv_dataset()
    print("Dataset csv type:", type(data2))
    print("Dataset csv structure:", data2)
    
    print("\n--- First 10 Elements of CSV ---")
    split2 = 'train' if 'train' in data2 else list(data2.keys())[0]
    
    for i in range(min(10, len(data2[split2]))):
        print(f"\nElement {i + 1}:")
        print(data2[split2][i])
