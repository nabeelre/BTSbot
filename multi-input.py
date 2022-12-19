import os

def main():
    models = ["vgg6_metadata_1_1", "vgg6_metadata_1_2", "vgg6_metadata_1_3",
              "vgg6_metadata_2_1", "vgg6_metadata_2_2", "vgg6_metadata_2_3",
              "vgg9_metadata_1_1"]
    
    for model in models:
        os.system(f"python bts_train.py {model} 1 750")
        os.system(f"python bts_train.py {model} 1 750")
        os.system(f"python bts_train.py {model} 1 750")

if __name__ == "__main__":
    main()