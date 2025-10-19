from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from astropy.table import Table
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import requests
import argparse
import io


def get_ps_image_table(ra, dec, filters="grizy"):
    """
    Query ps1filenames.py service to get a list of images

    ra, dec = position in degrees
    filters = string with filters to include. includes all by default
    Returns a table with the results

    Adapted from
    https://spacetelescope.github.io/mast_notebooks/notebooks/PanSTARRS/PS1_image/PS1_image.html
    """
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    # The final URL appends our query to the PS1 image service
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    # Read the ASCII table returned by the url
    table = Table.read(url, format='ascii')
    return table


def get_ps_url(ra, dec, size=252, im_format="jpeg", output_size=None):
    """
    Get URL for PanSTARRS images

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
    filters = string with filters to include. choose from "grizy"
    format = data format (options are "jpg", "png" or "fits")
    Returns a string with the URL

    Adapted from
    https://spacetelescope.github.io/mast_notebooks/notebooks/PanSTARRS/PS1_image/PS1_image.html
    """
    if output_size is None:
        output_size = size

    table = get_ps_image_table(ra, dec, filters="grizy")
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={im_format}&output_size={output_size}")

    if not all(f in table['filter'] for f in ['g', 'r', 'i']):
        print("One of g r and i is missing")
        return None

    flist = ["irgzy".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    table = table[np.isin(table['filter'], ['g', 'r', 'i'])]

    for i, param in enumerate(["red", "green", "blue"]):
        url = url + f"&{param}={table['filename'][i]}"

    return url


def download_image_batch(batch, survey):
    """Download images for a batch of sources from the specified survey"""
    results = []
    for source in batch:
        try:
            if survey == 'ls':
                # Legacy Survey query
                url = "https://www.legacysurvey.org/viewer/jpeg-cutout?" + \
                    f"ra={source['ra']}&dec={source['dec']}&" + \
                      "size=63&layer=ls-dr10&pixscale=1&bands=griy"

                response = requests.get(url)
                image = Image.open(io.BytesIO(response.content))
                image_array = np.array(image, dtype=np.float16)

                empty_image = all(32 == image_array.flatten())
                results.append((source['objectId'], image_array, empty_image))

            elif survey == 'ps':
                # PanSTARRS query
                url = get_ps_url(source['ra'], source['dec'], size=252, im_format="jpeg")
                if url is None:
                    results.append((source['objectId'], None, True))
                    continue

                r = requests.get(url)
                image = Image.open(io.BytesIO(r.content)).convert("RGB")
                image_array = np.array(image)

                # Bin to 1 arcsec/pixel resolution and normalize values to [0,1]
                image_array = image_array.reshape(63, 4, 63, 4, 3).mean(axis=(1, 3))
                image_array = image_array.astype(np.float32)
                image_array = image_array / image_array.max()

                results.append((source['objectId'], image_array, False))
            else:
                raise ValueError(f"Unknown survey: {survey}")

        except Exception as e:
            print(f"Error downloading image for objectId {source['objectId']}: {e}")
            results.append((source['objectId'], None, True))

    return results


def query_images(cand, survey, show_images=False, max_workers=None):
    """Query images for the given candidates from the specified survey"""
    img_cache = {}
    missing_col = f'missing_{survey.upper()}'  # "missing_PS" or "missing_LS"
    cand[missing_col] = False

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
            pool.imap(lambda batch: download_image_batch(batch, survey), batch_dicts),
            total=len(batch_dicts),
            desc=f"Downloading {survey.upper()} image batches",
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
                    cand.loc[cand['objectId'] == object_id, missing_col] = True

    return cand, img_cache


def process_dataset(survey, split_to_process, version, workers, show_images=False):
    """Process dataset for the specified survey, split, and version"""
    if split_to_process == "all":
        splits = ['train', 'val', 'test']
    else:
        splits = [split_to_process]

    for split in splits:
        print(f"Querying {survey.upper()} for {split} split...")

        # Load candidate data
        cand = pd.read_csv(f"data/{split}_cand_{version}_N100.csv", index_col=None)
        cand[['objectId', 'ra', 'dec']]

        # Query images based on survey
        cand, img_cache = query_images(cand, survey, show_images=show_images, max_workers=workers)
        missing_col = f'missing_{survey.upper()}'
        survey_suffix = f'{survey.upper()}63'

        # Create image array
        imgs = np.zeros((len(cand), 63, 63, 3), dtype=np.float16)
        for idx in cand.index:
            obj = cand.loc[idx]
            imgs[idx] = img_cache[obj['objectId']]

        # Save with missing images
        cand.to_csv(f"data/{split}_cand_{version}{survey_suffix}_N100.csv", index=False)
        np.save(f"data/{split}_triplets_{version}{survey_suffix}_N100.npy", imgs)

        print(f"ratio of missing {survey.upper()}: {np.sum(cand[missing_col])/len(cand)}")

        # Save without missing images
        imgs = imgs[~cand[missing_col]]
        cand = cand[~cand[missing_col]]

        cand.to_csv(f"data/{split}_cand_{version}{survey_suffix}nd_N100.csv", index=False)
        np.save(f"data/{split}_triplets_{version}{survey_suffix}nd_N100.npy", imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Query color cutouts from PanSTARRS or Legacy Survey'
    )
    parser.add_argument(
        '--survey', type=str, required=True, choices=['ps', 'ls'],
        help='Survey to query from: ps (PanSTARRS) or ls (Legacy Survey)'
    )
    parser.add_argument(
        '--split', type=str, default='train', choices=['train', 'val', 'test', 'all'],
        help='Dataset split to process (default: train)'
    )
    parser.add_argument(
        '--version', type=str, default='v11',
        help='Version identifier for the dataset (default: v11)'
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Number of workers to use for parallel processing (default: 8)'
    )
    parser.add_argument(
        '--show-images', action='store_true',
        help='Display images during processing (for debugging)'
    )

    args = parser.parse_args()

    process_dataset(
        survey=args.survey,
        split_to_process=args.split,
        version=args.version,
        workers=args.workers,
        show_images=args.show_images
    )
