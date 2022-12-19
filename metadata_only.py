import os

def main():
    models = ["metadata_only_1_1", "metadata_only_1_2", "metadata_only_1_3",
              "metadata_only_2_1", "metadata_only_2_2", "metadata_only_2_3",
              "metadata_only_3_1", "metadata_only_3_2", "metadata_only_3_3"]
    
    for model in models:
        os.system(f"python bts_train.py {model} 1 2500")
        os.system(f"python bts_train.py {model} 1 2500")
        os.system(f"python bts_train.py {model} 1 2500")

if __name__ == "__main__":
    main()