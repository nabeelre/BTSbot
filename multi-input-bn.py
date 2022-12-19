import os

def main():
    models = ["vgg6_metadata_1_bn_1", "vgg6_metadata_1_bn_2", "vgg6_metadata_1_bn_3",
              "vgg6_metadata_2_bn_1", "vgg6_metadata_2_bn_2", "vgg9_metadata_1_bn_1"]
    
    for model in models:
        os.system(f"python bts_train.py {model} 1 750")
        os.system(f"python bts_train.py {model} 1 750")
        os.system(f"python bts_train.py {model} 1 750")

if __name__ == "__main__":
    main()