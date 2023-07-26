import numpy as np, matplotlib.pyplot as plt, pandas as pd, time
import warnings, json, sys, requests, gzip, io, urllib, requests, os

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.colors import LogNorm

from penquins import Kowalski
from sklearn.model_selection import train_test_split

from bson.json_util import loads, dumps
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning 
warnings.filterwarnings("ignore", category=VerifyWarning)

import tqdm

BOLD = "\033[1m"; END  = "\033[0m"

with open('/Users/nabeelr/credentials.json', 'r') as f:
    creds = json.load(f)

k = Kowalski(username=creds['kowalski_username'], password=creds['kowalski_password'])
api_token = creds['fritz_api_key']
assert(k.ping())

external_HDD = "/Volumes/NRExternal3/trainv6 data/"
quest_raw_path = ""
to_desktop = "/Users/nabeelr/Desktop/"


def make_triplet(alert, normalize: bool = True):
    """
    Unpack binary fits files containing cutouts from kowalski
    Helper function to query_kowalski()

    Parameters
    ----------
    alert: dict
        alert dictionary queried from kowlaski
        see query_kowalski()
    
    normalize (optional): bool
        normalize cutouts by the Frobenius norm (L2) 

    Returns
    -------
    triplet: 63 x 63 x 3 array
        3 channel 63 x 63 image representing the science, reference, and difference cutouts
    
    drop: bool
        whether or not the file is found to be corrupted
    
    ----------------------------------------------------------
    ADAPTED FROM https://github.com/dmitryduev/braai
    """
    
    cutout_dict = dict()
    drop = False
    
    for cutout in ('science', 'template', 'difference'):
        cutout_data = loads(dumps([alert[f'cutout{cutout.capitalize()}']['stampData']]))[0]
        # unzip fits file
        with gzip.open(io.BytesIO(cutout_data), 'rb') as f:
            with fits.open(io.BytesIO(f.read())) as hdu:
                data = hdu[0].data

                # Compute median value of image to fill nans
                medfill = np.nanmedian(data.flatten())
                
                # if the median is not a typical pixel value, image is corrupted; mark to be excluded
                if medfill == np.nan or medfill == -np.inf or medfill == np.inf:
                    print(alert['objectId'], "bad medfill (nan or inf)", alert['candidate']['candid'])
                    drop = True
                
                # Fill in nans with median value
                cutout_dict[cutout] = np.nan_to_num(data, nan=medfill)
                
                # normalize with L2 norm
                if normalize and not drop:
                    cutout_dict[cutout] /= np.linalg.norm(cutout_dict[cutout])
                    
                # If image is all zeros, image is corrupted; mark to be excluded
                if np.all(cutout_dict[cutout].flatten() == 0):
                    print(alert['objectId'], "zero image", alert['candidate']['candid'])
                    drop=True
                
                # If any nans remain in image, image is corrupted; mark to be excluded
                # Should never trigger because nans were already filled
#                 if np.any(np.isnan(cutout_dict[cutout].flatten())):
#                     print(alert['objectId'], "nan here", alert['candid'])
#                     drop=True
                    
        # pad to 63x63 if smaller
        shape = cutout_dict[cutout].shape
        if shape != (63, 63):
            print("bad shape", shape, alert['candidate']['candid'], alert['objectId'])
            # Fill value will have changed after normalizing so recompute
            medfill = np.nanmedian(cutout_dict[cutout].flatten())
            
            # Execute padding
            cutout_dict[cutout] = np.pad(cutout_dict[cutout],
                                         [(0, 63 - shape[0]),
                                          (0, 63 - shape[1])],
                                          mode='constant', 
                                          constant_values=medfill)
            
    triplet = np.zeros((63, 63, 3))
    triplet[:, :, 0] = cutout_dict['science']
    triplet[:, :, 1] = cutout_dict['template']
    triplet[:, :, 2] = cutout_dict['difference']
    
    return triplet, drop


def query_kowalski(ZTFID, kowalski, programid, normalize : bool = True, verbose : bool = False, save_raw = None, load_raw = None):
    """
    Query kowalski for alerts with cutouts for a (list of) ZTFID(s)

    Parameters
    ----------
    ZTFID: string or list
        Object IDs to query for (e.g. ZTF22abwqedu)
    
    kowalski:
        a kowalski api object created with the kowalski library
        
    normalize (optional): bool
        normalize cutouts by the Frobenius norm (L2)
        
    programid:
        which program to pull alerts from (1=public, 2=collab, 3=caltech mode)
        
    verbose (optional): bool
        print diagnostics after each query
        
    save_raw (optional): str
        if provided, all query results will be individually saved to disk at this path before any processsing is done
        
    load_raw (optional): str
        if provided, check for existing file at this path before querying, load file and continue processing as if just queried

    Returns
    -------
    alerts: list of dicts
        each dict represents alert
        alert columns include jd, ra, dec, candid, acai and braii scores, magpsf, cutouts, etc.
        
    
    ADAPTED FROM https://github.com/growth-astro/ztfrest/
    https://zwickytransientfacility.github.io/ztf-avro-alert/schema.html
    """
    
    # Deal with provided input being a single ZTF object (string) and multiple (list)
    if type(ZTFID) == str:
        list_ZTFID = [ZTFID]
    elif type(ZTFID) == list:
        list_ZTFID = ZTFID
    else:
        print(f"{ZTFID} must be a list or a string")
        return None

    alerts = []
    
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
                                        
                    "classifications.acai_h": 1,
                    "classifications.acai_v": 1,
                    "classifications.acai_o": 1,
                    "classifications.acai_n": 1,
                    "classifications.acai_b": 1,
                    
                    "cutoutScience": 1,
                    "cutoutTemplate": 1,
                    "cutoutDifference": 1,
                }
            }
        }
        
        object_alerts = None
        existing_data_path = None
        
        # Check if file path is provided for locating preloaded data
        if type(load_raw) == str:
            existing_data_path = os.path.join(load_raw, f"{ZTFID}_prog{programid}.npy")
            
            if os.path.exists(existing_data_path):
                # Read existing data
                object_alerts = np.load(existing_data_path, allow_pickle=True)
                print(f"    loaded existing data for {ZTFID}")
            else:
                print(f"    could not find existing data for {ZTFID}")
                existing_data_path = None
        
        # if opting to not use preloaded data or preloaded data couldn't be found
        if object_alerts is None:
            # Execute query
            r = kowalski.query(query)
            
            if r['data'] == []:
                # No alerts recieved - possibly by failed query (connection or permissions)
                print(f"  No programid={programid} data for", ZTFID)
                continue
            else:
                # returned data is list of dicts, each dict is an alert packet
                object_alerts = r['data']   

        # Only try to save raw data if preloaded data couldn't be found
        if existing_data_path is None:
            if type(save_raw) == str:
                if not os.path.exists(save_raw):
                    os.makedirs(save_raw)
                np.save(os.path.join(save_raw, f"{ZTFID}_prog{programid}"), object_alerts)
            elif save_raw is not None:
                print(f"Could not find save directory: {save_raw}")
                print("No queries will be saved")
                save_raw = None

        # initialize empty array to contain triplets
        triplets = np.empty((len(object_alerts), 63, 63, 3))
        # some images will be corrupted, initialize array to log which to exclude
        to_drop = np.array((), dtype=int)

        # For each alert ...
        for i, alert in enumerate(object_alerts):
            # Unzip fits files of cutouts
            triplets[i], drop = make_triplet(alert, normalize=normalize)

            # Note the alert/triplet index where a cutout was found to be corrupted 
            if drop:
                to_drop = np.append(to_drop, int(i))

        # Delete corresponding triplets and alerts that had corrupted cutouts
        if len(to_drop) > 0:
            triplets = np.delete(triplets, list(to_drop), axis=0)
            object_alerts = np.delete(object_alerts, list(to_drop), axis=0)

        # candidate and classifications are two dicts nested within the alert dict
        # This merges those two dicts and does away with the unncessary data in the alert dict 
        # object_alerts = [alert['candidate'] | alert['classifications'] for alert in object_alerts]

        # Add triplet to the alert dict
        for alert, triplet in zip(object_alerts, triplets):
            alert['triplet'] = triplet

        alerts += list(object_alerts)

        if verbose:
            print(f"  Finished {'loading' if existing_data_path else 'querying'}", ZTFID)
    
    if verbose:
        print(f"Finished all programid={programid} queries, got {len(alerts)} alerts\n")
    
    return alerts


def extract_triplets(alerts, normalize: bool = True, pop_triplet: bool = True):
    """
    Takes in alerts (list of dicts) with key 'triplet', pops triplets out of alerts, 
    and returns alerts and triplets separated
    """
    triplets = np.empty((len(alerts), 63, 63, 3))
    for i, alert in enumerate(alerts):
        triplets[i] = alert['triplet']
        
        if pop_triplet:
            alert.pop('triplet'); alert.pop('cutoutScience'); alert.pop('cutoutTemplate'); alert.pop('cutoutDifference')
        
    return alerts, triplets


def prep_alerts(alerts, label):
    """
    takes in alerts (list of dicts) with nested dicts 'candidate' and 'classifications'
    un-nests inner dicts and adds column containing provided labels
    returns dataframe 
    """
    cand_class_data = [alert['candidate'] | alert['classifications'] for alert in alerts]

    df = pd.DataFrame(cand_class_data)
    df.insert(0, "objectId", [alert['objectId'] for alert in alerts])
#     df.insert(1, "candid", [alert['candid'] for alert in alerts])
    
    # label must be int equalling 0, 1 or a list of 1s and 0s
    if type(label) == list or type(label) == np.ndarray:
        assert(len(label) == len(alerts))
        df.insert(2, "label", label)
    elif type(label) == int:    
        df.insert(2, "label", np.full((len(alerts),), label, dtype=int))
    print("Arranged candidate data and inserted labels")
    return df


def download_training_data(source_df, set_name, kowalski, label, normalize_cutouts : bool = True, verbose : bool = False, save_raw = None, load_raw = None):
    """
    Downloads alerts with cutouts from kowalski
    Saves triplets in a .npy and alert metadata in a .csv
    
    Parameters
    ----------
    source_df: dataframe
        dataframe with columns "ZTFID"
    
    kowalski:
        a kowalski api object created with the penquins library
        
    label: int, array_like, or "compute"
        BTS / not BTS label to assign to each alert in saved csv
        if int (must be 0 or 1) assign all alerts provided label
        if array_like (length must match number of alerts) assign from array in order
        if "compute" assign all objects with any alert with magpsf < 18.5 label=1, otherwise 0
        
    normalize_cutouts (optional)- see query_kowalski()
        
    verbose (optional): bool
        print diagnostics
        
    save_raw, load_raw (optional) - see query_kowalski()

    Returns
    -------
    Nothing
    """
    
    if verbose:
        print(f"Querying kowalski for {len(source_df['ZTFID'])} objects of {set_name}")
        
    alerts, triplets = extract_triplets(query_kowalski(source_df['ZTFID'].to_list(), k, 1, normalize=normalize_cutouts, 
                                                       verbose=verbose, save_raw=save_raw, load_raw=load_raw) + 
                                        query_kowalski(source_df['ZTFID'].to_list(), k, 2, normalize=normalize_cutouts, 
                                                       verbose=verbose, save_raw=save_raw, load_raw=load_raw))

    np.save(f"data/base_data/{set_name}_triplets.npy", triplets)
    del triplets
    print("Saved and purged triplets\n")

    num_alerts = len(alerts)
    
    if type(label) == int:
        label = np.full((num_alerts), label, dtype=int)
    elif type(label) == list or type(label) == np.ndarray:
        label = label
    elif label == "compute":
        true_objs = set()
        for alert in alerts: 
            if alert['candidate']['magpsf'] < 18.5:
                true_objs.add(alert['objectId'])
        label = np.asarray([1 if alert['objectId'] in true_objs else 0 for alert in alerts])
    else:
        print(f"Could not understand label: {label}")
    
    num_trues = np.sum(label == 1)
    num_falses = np.sum(label == 0)
    if num_trues + num_falses != len(label):
        print(f"Invalid labels provided: {label}")
    else:
        print(f"{set_name} {len(label)} total alerts: {num_trues} trues, {num_falses} falses")

    cand_data = prep_alerts(alerts, label)
    cand_data.to_csv(f'data/base_data/{set_name}_candidates.csv', index=False)
    del cand_data
    print("Saved and purged candidate data")
    

if __name__ == "__main__":
    set = sys.argv[1]

    set_df = pd.read_csv("data/base_data/trues_cleaned.csv", index_col=None)

    if set == "trues":
        label = 1
    elif set in ["dims", "vars", "rejects"]:
        label = 0
    elif set == "extIas":
        label = "compute"
    else:
        print(set)
        exit()

    download_training_data(set_df, set, k, label=label, normalize_cutouts=True, verbose=True, save_raw=external_HDD+set, load_raw=external_HDD+set)
