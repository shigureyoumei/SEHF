import h5py

def print_h5_contents(file_path):
    def print_attrs(name, obj):
        print(f"{name}:")
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")
        if isinstance(obj, h5py.Group):
            print("    (Group)")
        elif isinstance(obj, h5py.Dataset):
            print(f"    (Dataset) - shape: {obj.shape}, dtype: {obj.dtype}")

    with h5py.File(file_path, 'r') as f:
        f.visititems(print_attrs)

# 使用示例
file_path = '/mnt/e/Program/PROJECT/dataset/DATASETS/try6/try6.h5'
print_h5_contents(file_path)