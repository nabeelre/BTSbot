import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset, Features, Array3D, Value


class FlexibleDataset(TorchDataset):
    def __init__(self, images=None, metadata=None, labels=None, transform=None):
        self.images = images
        self.metadata = metadata
        self.labels = labels
        self.transform = transform
        self.need_triplets = images is not None
        self.need_metadata = metadata is not None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_item = self.labels[idx]
        image_item = None
        meta_item = None

        if self.need_triplets:
            image_item = self.images[idx]
            if self.transform:
                image_item = self.transform(image_item)

        if self.need_metadata:
            meta_item = self.metadata[idx]

        if self.need_triplets and self.need_metadata:
            return image_item, meta_item, label_item
        elif self.images is not None:
            return image_item, label_item
        elif self.metadata is not None:
            return meta_item, label_item


class RandomRightAngleRotation(object):
    def __call__(self, img):
        degrees = np.random.choice([0, 90, 180, 270])
        return transforms.functional.rotate(img, degrees)


def make_report(config, report_path, run_data, val_summ):
    # generate training report in json format
    print('Generating report...', end='')
    report = {
        'Run time stamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'Run name': run_data['run_name'],
        'Training history': {k: v for k, v in run_data.items() if k != 'run_name'},
        'train_config': dict(config),
        'val_summary': dict(val_summ),
    }
    for k in report['Training history'].keys():
        report['Training history'][k] = np.array(report['Training history'][k]).tolist()

    f_name = os.path.join(report_path)
    with open(f_name, 'w') as f:
        json.dump(report, f, indent=4)
    print('done')


def save_model(model, image_size, path: str):
    """
    Save pytorch model to disk in specified format

    Parameters
    ----------
    model:
        Pytorch model

    model_dir: str
        Directory to save model to

    kind: str
        Format to write model as, options: "pth", "ONNX", "all"
    """

    # if kind == "pth" or kind == "all":
    torch.save(model.state_dict(), path)

    # if kind == "scripted" or kind == "all":
    #     try:
    #         model_without_domain_clf = DANN(config)
    #         model_without_domain_clf.load_state_dict(model.state_dict())
    #         model_without_domain_clf.eval()
    #         model_scripted = torch.jit.script(model_without_domain_clf)
    #     except Exception as e:
    #         print("Failed to convert model to scripted format")
    #         print(e)
    #         return
    #     model_scripted.save(f"{DANN_DIR}/{model_dir}/best_model.pt")

    # if kind == "ONNX" or kind == "all":
    #     example_input = torch.randn(1, 3, image_size, image_size).to(device)

    #     model.eval()
    #     torch.onnx.export(
    #         model, example_input, f'{DANN_DIR}/{model_dir}/best_model.onnx',
    #         verbose=False, input_names=["triplet"], output_names=["RB"]
    #     )
    #     model.train()

    return


def convert_to_hf(split, version):
    triplets = np.load(f"data/{split}_triplets_{version}_N100.npy")  # shape: (N, 63, 63, 3)
    cand = pd.read_csv(f"data/{split}_cand_{version}_N100.csv")

    # cand['triplet'] = list(triplets)

    feature_types = {
        "triplet": Array3D(dtype="float32", shape=(63, 63, 3)),
    }

    for col in cand.columns:
        if col == "candid":
            feature_types[col] = Value("string")
        elif col != "triplet":
            if cand[col].dtype == object:
                feature_types[col] = Value("string")
            elif np.issubdtype(cand[col].dtype, np.bool_):
                feature_types[col] = Value("bool")
            elif cand[col].dtype == int:
                feature_types[col] = Value("int32")
            elif cand[col].dtype == float:
                feature_types[col] = Value("float32")
            else:
                print(f"Unknown dtype for column {col}: {cand[col].dtype}")
            # print(col, np.max(cand[col]), np.min(cand[col]))
    # print(feature_types)

    data_dict = cand.to_dict(orient="list")
    data_dict["triplet"] = list(triplets)
    dataset = Dataset.from_dict(data_dict, features=Features(feature_types))
    # dataset = Dataset.from_pandas(cand, features=Features(feature_types), preserve_index=False)

    dataset.save_to_disk(f"data/{split}_{version}_N100")
