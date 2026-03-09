from conn import load_connections_from_hf

def train_test_split(test_size=0.2,seed=175):
    """
    Returns ds_train, ds_test splits from Hugging Face
    """
    ds = load_connections_from_hf()
    split = ds.train_test_split(test_size=test_size, seed=seed)
    ds_train = split["train"]
    ds_test = split["test"]
    return ds_train, ds_test
