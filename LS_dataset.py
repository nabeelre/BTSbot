import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import requests
import io
import argparse
from multiprocessing import Pool, cpu_count


def download_image_batch(batch):
    """Download images for a batch of sources"""
    results = []
    for source in batch:
        try:
            url = "https://www.legacysurvey.org/viewer/jpeg-cutout?" + \
                f"ra={source['ra']}&dec={source['dec']}&" + \
                  "size=224&layer=ls-dr10&pixscale=0.25&bands=griy"

            response = requests.get(url)
            image = Image.open(io.BytesIO(response.content))
            image_array = np.array(image)

            empty_image = all(32 == image_array.flatten())
            results.append((source['objectId'], image_array, empty_image))
        except Exception as e:
            print(f"Error downloading image for objectId {source['objectId']}: {e}")
            results.append((source['objectId'], None, True))

    return results


def build_LS_image_cache(cand, show_images=False, max_workers=None):
    img_cache = {}
    cand['missing_LS'] = False

    objs = cand[['objectId', 'ra', 'dec']].drop_duplicates('objectId')
    print(len(objs), "unique objects")

    if max_workers is None:
        max_workers = min(cpu_count(), len(objs))
    print("Using", max_workers, "workers")

    with Pool(processes=max_workers) as pool:
        # Split the dataframe into batches for parallel processing
        batch_size = max(1, len(objs) // (3 * max_workers))
        batches = [objs.iloc[i:i + batch_size] for i in range(0, len(objs), batch_size)]

        # Convert batches to list of dictionaries for easier processing
        batch_dicts = [batch.to_dict('records') for batch in batches]

        # Process batches in parallel
        results = list(tqdm(
            pool.imap(download_image_batch, batch_dicts),
            total=len(batch_dicts),
            desc="Downloading image batches",
            unit="batch"
        ))

        # Flatten the list of lists into a single list
        results = [item for sublist in results for item in sublist]

        # Process results
        for object_id, image_array, is_empty in results:
            if image_array is not None:
                img_cache[object_id] = image_array

                if show_images:
                    plt.imshow(image_array)
                    plt.show()

                if is_empty:
                    cand.loc[cand['objectId'] == object_id, 'missing_LS'] = True

    return cand, img_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process LS dataset with specified split and version'
    )
    parser.add_argument(
        '--split', type=str, default='train', choices=['train', 'val', 'test'],
        help='Dataset split to process (default: train)'
    )
    parser.add_argument(
        '--version', type=str, default='v11',
        help='Version identifier for the dataset (default: v11)'
    )

    args = parser.parse_args()
    split = args.split
    version = args.version

    cand = pd.read_csv(f"data/{split}_cand_{version}_N100.csv", index_col=None)
    cand[['objectId', 'ra', 'dec']]

    cand, img_cache = build_LS_image_cache(cand, show_images=False, max_workers=8)

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
