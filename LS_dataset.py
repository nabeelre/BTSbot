import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import requests
import io


def build_LS_image_cache(cand, show_images=False):
    img_cache = {}
    cand['missing_LS'] = False

    objs = cand[['objectId', 'ra', 'dec']].drop_duplicates('objectId')
    print(len(objs), "unique objects")

    for idx in tqdm(objs.index):
        source = objs.loc[idx]

        if source['objectId'] in img_cache:
            continue

        url = "https://www.legacysurvey.org/viewer/jpeg-cutout?" + \
            f"ra={source['ra']}&dec={source['dec']}&" + \
              "size=224&layer=ls-dr10&pixscale=0.25&bands=griy"

        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        image_array = np.array(image)
        if show_images:
            plt.imshow(image_array)
            plt.show()

        img_cache[source['objectId']] = image_array

        if all(32 == image_array.flatten()):
            # print("empty image")
            cand.loc[cand['objectId'] == source['objectId'], 'missing_LS'] = True

    return cand, img_cache


if __name__ == "__main__":
    split = "test"
    version = "v11"

    cand = pd.read_csv(f"data/{split}_cand_{version}_N100.csv", index_col=None)
    cand[['objectId', 'ra', 'dec']]

    cand, img_cache = build_LS_image_cache(cand, show_images=False)

    LS_imgs = np.zeros((len(cand), 224, 224, 3))
    for idx in cand.index:
        obj = cand.loc[idx]
        LS_imgs[idx] = img_cache[obj['objectId']]

    cand.to_csv(f"data/{split}_cand_{version}LS_N100.csv", index=False)
    np.save(f"data/{split}_triplets_{version}LS_N100.npy", LS_imgs)

    print(f"ratio of missing LS: {np.sum(cand['missing_LS'])/len(cand)}")

    LS_imgs = LS_imgs[~cand['missing_LS']]
    cand = cand[~cand['missing_LS']]

    cand.to_csv(f"data/{split}_cand_{version}LSnd_N100.csv", index=False)
    np.save(f"data/{split}_triplets_{version}LSnd_N100.npy", LS_imgs)
