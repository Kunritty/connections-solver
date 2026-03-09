from datasets import load_dataset

def load_huggingface_dataset():
    dataset = load_dataset("tm21cy/NYT-Connections")
    return dataset

def load_csv_dataset():
    pass

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
