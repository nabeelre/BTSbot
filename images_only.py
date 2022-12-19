import os

def main():
    models = ["vgg6_1", "vgg6_2", "vgg6_3", "vgg6_4", "vgg9_1", "vgg9_2"]
    
    for model in models:
        os.system(f"python bts_train.py {model} 1 500")
        os.system(f"python bts_train.py {model} 1 500")
        os.system(f"python bts_train.py {model} 1 500")

if __name__ == "__main__":
    main()