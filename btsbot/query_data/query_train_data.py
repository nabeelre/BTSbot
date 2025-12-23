import pandas as pd
import numpy as np
import sys
import os

from alert_utils import (make_triplet, extract_triplets, rerun_braai,
                         prep_alerts, crop_triplets)
from compile_ZTFIDs import compile_ZTFIDs

external_HDD = "/Volumes/NRExternal3/trainv8 data/"
quest_raw_path = "v12raw/"
to_desktop = "/Users/nabeelr/Desktop/"

KOWALSKI_USER = os.environ.get('KOWALSKI_USER')
KOWALSKI_PASS = os.environ.get('KOWALSKI_PASS')

if KOWALSKI_USER is not None and KOWALSKI_PASS is not None:
    from penquins import Kowalski

    k = Kowalski(instances={
        'kowalski': {
            'protocol': 'https',
            'port': 443,
            'host': 'kowalski.caltech.edu',
            'username': KOWALSKI_USER,
            'password': KOWALSKI_PASS
        }
    })
else:
    print("No Kowalski credentials found. Querying Kowalski will not be possible.")
    k = None


def query_kowalski(ZTFID, kowalski, programid, include_cutouts: bool = True,
                   normalize: bool = True, verbose: bool = False, save_raw=None, load_raw=None):
    """
    Query kowalski for alerts with cutouts for a (list of) ZTFID(s)

    Parameters
    ----------
    ZTFID: string or list
        Object IDs to query for (e.g. ZTF22abwqedu)

    kowalski:
        a kowalski api object created with the kowalski library

    include_cutouts (optional): bool
        Easy flag for including/excluding cutouts from query

    normalize (optional): bool
        normalize cutouts by the Frobenius norm (L2)

    programid:
        which program to pull alerts from (1=public, 2=collab, 3=caltech mode)

    verbose (optional): bool
        print diagnostics after each query

    save_raw (optional): str
        if provided, all query results will be individually saved to disk at
        this path before any processsing is done

    load_raw (optional): str
        if provided, check for existing file at this path before querying, load
        file and continue processing as if just queried

    Returns
    -------
    alerts: list of dicts
        each dict represents alert
        alert columns include jd, ra, dec, candid, acai and braii scores,
        magpsf, cutouts, etc.

    Adapted from: https://github.com/growth-astro/ztfrest/
    See here for ZTF alert packet feature definitions:
        https://zwickytransientfacility.github.io/ztf-avro-alert/schema.html

    This can also be done by querying from Fritz instead of Kowalski.
    """

    # Deal with input being a single ZTF object (string) and multiple (list)
    if isinstance(ZTFID, str):
        list_ZTFID = [ZTFID]
    elif isinstance(ZTFID, list):
        list_ZTFID = ZTFID
    else:
        print(f"{ZTFID} must be a list or a string")
        return None

    alerts = []

    if include_cutouts:
        cutout_query_dict = {
            "cutoutScience": 1,
            "cutoutTemplate": 1,
            "cutoutDifference": 1
        }
    else:
        cutout_query_dict = {}

    # For each object requested ...
    for ZTFID in list_ZTFID:
        # Set up query
        query = {
            "query_type": "find",
            "query": {
                "catalog": "ZTF_alerts",
                "filter": {
                    # take only alerts for specified object
                    'objectId': ZTFID,
                    # take only alerts with specified programid
                    "candidate.programid": programid,
                },
                # what quantities to recieve
                "projection": {
                    "_id": 0,
                    "objectId": 1,

                    "candidate.candid": 1,
                    "candidate.programid": 1,
                    "candidate.fid": 1,
                    "candidate.isdiffpos": 1,
                    "candidate.ndethist": 1,
                    "candidate.ncovhist": 1,
                    "candidate.sky": 1,
                    "candidate.fwhm": 1,
                    "candidate.seeratio": 1,
                    "candidate.mindtoedge": 1,
                    "candidate.nneg": 1,
                    "candidate.nbad": 1,
                    "candidate.scorr": 1,
                    "candidate.dsnrms": 1,
                    "candidate.ssnrms": 1,
                    "candidate.exptime": 1,

                    "candidate.field": 1,
                    "candidate.jd": 1,
                    "candidate.ra": 1,
                    "candidate.dec": 1,

                    "candidate.magpsf": 1,
                    "candidate.sigmapsf": 1,
                    "candidate.diffmaglim": 1,
                    "candidate.magap": 1,
                    "candidate.sigmagap": 1,
                    "candidate.magapbig": 1,
                    "candidate.sigmagapbig": 1,
                    "candidate.magdiff": 1,
                    "candidate.magzpsci": 1,
                    "candidate.magzpsciunc": 1,
                    "candidate.magzpscirms": 1,

                    "candidate.distnr": 1,
                    "candidate.magnr": 1,
                    "candidate.sigmanr": 1,
                    "candidate.chinr": 1,
                    "candidate.sharpnr": 1,

                    "candidate.neargaia": 1,
                    "candidate.neargaiabright": 1,
                    "candidate.maggaia": 1,
                    "candidate.maggaiabright": 1,

                    "candidate.drb": 1,
                    "candidate.classtar": 1,
                    "candidate.sgscore1": 1,
                    "candidate.distpsnr1": 1,
                    "candidate.sgscore2": 1,
                    "candidate.distpsnr2": 1,
                    "candidate.sgscore3": 1,
                    "candidate.distpsnr3": 1,

                    "candidate.jdstarthist": 1,
                    "candidate.jdstartref": 1,

                    "candidate.sgmag1": 1,
                    "candidate.srmag1": 1,
                    "candidate.simag1": 1,
                    "candidate.szmag1": 1,

                    "candidate.sgmag2": 1,
                    "candidate.srmag2": 1,
                    "candidate.simag2": 1,
                    "candidate.szmag2": 1,

                    "candidate.sgmag3": 1,
                    "candidate.srmag3": 1,
                    "candidate.simag3": 1,
                    "candidate.szmag3": 1,

                    "candidate.nmtchps": 1,
                    "candidate.clrcoeff": 1,
                    "candidate.clrcounc": 1,
                    "candidate.chipsf": 1,

                    "classifications.acai_h": 1,
                    "classifications.acai_v": 1,
                    "classifications.acai_o": 1,
                    "classifications.acai_n": 1,
                    "classifications.acai_b": 1,
                    "classifications.bts": 1,
                } | cutout_query_dict
            }
        }

        object_alerts = None
        load_path = None

        # Check if file path is provided for locating preloaded data
        if isinstance(load_raw, str):
            load_path = os.path.join(load_raw, f"{ZTFID}_prog{programid}.npy")

            if os.path.exists(load_path):
                # Read existing data
                object_alerts = np.load(load_path, allow_pickle=True)
                print(f"Loaded existing data for {ZTFID}")
            else:
                print(f"Could not find existing data for {ZTFID}")
                load_path = None

        # if not use preloaded data or preloaded data couldn't be found
        if object_alerts is None:
            # Execute query
            r = kowalski.query(query)

            if r['kowalski']['data'] == []:
                # No alerts recieved - possibly due to connection or permissions
                print(f"  No programid={programid} data for", ZTFID)
                continue
            else:
                # returned data is list of dicts, each dict is an alert packet
                object_alerts = r['kowalski']['data']

        # Only try to save raw data if preloaded data couldn't be found
        if load_path is None:
            if isinstance(save_raw, str):
                if not os.path.exists(save_raw):
                    os.makedirs(save_raw)
                np.save(os.path.join(save_raw, f"{ZTFID}_prog{programid}"),
                        object_alerts)
            elif save_raw is not None:
                print(f"Could not find save directory: {save_raw}")
                print("No queries will be saved")
                save_raw = None

        if include_cutouts:
            # initialize empty array to contain triplets
            triplets = np.empty((len(object_alerts), 63, 63, 3))
            # some images corrupted, initialize array to log which to exclude
            to_drop = np.array((), dtype=int)

            # For each alert ...
            for i, alert in enumerate(object_alerts):
                # Unzip fits files of cutouts
                triplets[i], drop = make_triplet(alert, normalize=normalize)

                # Note the index where a cutout was found to be corrupted
                if drop:
                    to_drop = np.append(to_drop, int(i))

            # Delete corresponding triplets and alerts that had corrupted cutouts
            if len(to_drop) > 0:
                triplets = np.delete(triplets, list(to_drop), axis=0)
                object_alerts = np.delete(object_alerts, list(to_drop), axis=0)

            # Add triplet to the alert dict
            for alert, triplet in zip(object_alerts, triplets):
                alert['triplet'] = triplet

        alerts += list(object_alerts)

        if verbose:
            print(f"  Finished {'loading' if load_path else 'querying'}", ZTFID)

    if verbose:
        print(f"\nFinished all programid={programid} queries",
              f"got {len(alerts)} alerts\n\n")

    return alerts


def download_training_data(query_df, query_name, label,
                           include_cutouts: bool = True,
                           normalize_cutouts: bool = True,
                           cutout_size=63,
                           verbose: bool = False,
                           save_raw=None, load_raw=None):
    """
    Downloads alerts with cutouts from kowalski for query with query_name and
    list of ZTFIDs stored in query_df
    Saves triplets in a .npy and alert metadata in a .csv

    Parameters
    ----------
    query_df: DataFrame
        dataframe with column "ZTFID"

    query_name: str
        name of query

    label: int, array_like, or "compute"
        BTS / not BTS label to assign to each alert in saved csv
        if int (must be 0 or 1) assign all alerts provided label
        if array_like (length must match number of alerts) assign from array in order
        if "compute" assign all objects with any alert with magpsf < 18.5 label=1, otherwise 0

    normalize_cutouts (optional) - see query_kowalski()

    verbose (optional): bool

    save_raw, load_raw (optional) - see query_kowalski()

    Returns
    -------
    Nothing
    """

    if verbose:
        print(f"Querying kowalski for {len(query_df)} objects of {query_name}")

    if k is not None and k.ping('kowalski'):
        print("Connected to Kowalski")
    else:
        print("Unable to connect to Kowalski")
        exit()

    # Query programid=1 and 2 alerts from kowalski for all ZTFIDs and separate
    # their triplets from the rest of their alert packets
    query_response = query_kowalski(
        query_df['ZTFID'].to_list(), k, programid=1,
        include_cutouts=include_cutouts, normalize=normalize_cutouts,
        verbose=verbose, save_raw=save_raw, load_raw=load_raw
    ) + query_kowalski(
        query_df['ZTFID'].to_list(), k, programid=2,
        include_cutouts=include_cutouts, normalize=normalize_cutouts,
        verbose=verbose, save_raw=save_raw, load_raw=load_raw
    )

    if include_cutouts:
        alerts, triplets = extract_triplets(query_response)
    else:
        alerts = query_response

    num_alerts = len(alerts)

    # Turn provided label into array of length num_alerts
    if isinstance(label, int):
        label = np.full((num_alerts), label, dtype=int)
    elif isinstance(label, int) or isinstance(label, np.ndarray):
        label = label
    elif label == "compute":
        true_objs = set()
        for alert in alerts:
            if alert['candidate']['magpsf'] < 18.5:
                true_objs.add(alert['objectId'])
        label = np.asarray([1 if alert['objectId'] in true_objs else 0 for alert in alerts])
    else:
        print(f"Could not understand label: {label}")
        label = np.full((num_alerts), None)

    if None not in label:
        num_trues = np.sum(label == 1)
        num_falses = np.sum(label == 0)
        if num_trues + num_falses == len(label):
            print(f"{query_name} {len(label)} total alerts:",
                  f"{num_trues} trues, {num_falses} falses")

    if include_cutouts:
        # Rerun braai on all triplets and store their scores to be added to metadata
        new_drb = rerun_braai(triplets)

        # Optionally, crop and renormalize all cutouts
        if cutout_size != 63:
            triplets = crop_triplets(triplets, cutout_size)

        # Save triplets to disk and purge from memory
        np.save(f"data/base_data/{query_name}_triplets" +
                f"{cutout_size if cutout_size != 63 else ''}.npy", triplets)
        del triplets
        print("Saved and purged triplets\n")
    else:
        new_drb = -1

    # augment alerts with custom features and add in labels
    cand_data = prep_alerts(alerts, label, new_drb)

    # Save metadata to disk and purge from memory
    cand_data.to_csv(f'data/base_data/{query_name}_candidates.csv', index=False)
    del cand_data
    print("Saved and purged candidate data")


if __name__ == "__main__":
    query_name = sys.argv[1]

    # if file of query's ZTFIDs doesn't exist, run compile_ZTFIDs
    if not os.path.exists(f"data/base_data/{query_name}.csv"):
        compile_ZTFIDs()

    query_df = pd.read_csv(f"data/base_data/{query_name}.csv", index_col=None)

    if query_name == "trues":
        label = 1
    elif query_name in ["dims", "vars", "rejects", "junk", "extra_agn", "extra_cvs"]:
        label = 0
    elif query_name == "extIas":
        label = "compute"
    else:
        print(query_name, "not known")
        exit()

    download_training_data(
        query_df, query_name, label=label,
        normalize_cutouts=True, include_cutouts=False, verbose=True,
        save_raw=quest_raw_path + query_name,
        load_raw=quest_raw_path + query_name
    )
