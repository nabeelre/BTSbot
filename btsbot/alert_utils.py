from bson.json_util import loads, dumps
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from astropy.io import fits
# import tensorflow as tf
import pandas as pd
import numpy as np
import tqdm
import gzip
import io
import os

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
    k = None


def plot_triplet(trip):
    """Adapted from https://github.com/dmitryduev/braai"""
    fig = plt.figure(figsize=(8, 2), dpi=120)

    ax1 = fig.add_subplot(131)
    ax1.axis('off')
    ax1.imshow(trip[:, :, 0], origin='upper', cmap=plt.cm.bone, norm=LogNorm())
    ax1.title.set_text('Science')

    ax2 = fig.add_subplot(132)
    ax2.axis('off')
    ax2.imshow(trip[:, :, 1], origin='upper', cmap=plt.cm.bone, norm=LogNorm())
    ax2.title.set_text('Reference')

    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    ax3.imshow(trip[:, :, 2], origin='upper', cmap=plt.cm.bone)
    ax3.title.set_text('Difference')

    return fig


def crop_norm_cutout(cutout, crop_to_size):
    """
    Crop 63x63 cutout to crop_to_size x crop_to_size pixels and normalize with
    L2-norm

    Parameters
    ----------
    cutout: array
        63x63 pixel array representing image cutout

    crop_to_size: int
        Integer representing desired image length/width

    Returns
    -------
    cutout: array
        image cutout cropped to crop_to_size x crop_to_size pixels and normalized
    """

    margin = (63 - crop_to_size) // 2

    cutout = cutout[margin:margin + crop_to_size, margin:margin + crop_to_size]
    cutout /= np.linalg.norm(cutout)

    return cutout


def crop_triplets(triplets, crop_to_size):
    """
    Crop all cutouts in array of triplets to crop_to_size x crop_to_size
    and renormalize with L2-norm

    Parameters
    ----------
    triplets: array
        Array of triplets whose cutouts are to be cropped

    crop_to_size: int
        Integer representing desired image length/width

    Returns
    -------
    triplets: array
        Array of triplets each cutout has been cropped and renormalized
    """

    cropped_triplets = np.zeros((len(triplets), crop_to_size, crop_to_size, 3))

    for trip_i in range(len(triplets)):
        for cut_i in range(3):
            cropped_triplets[trip_i, :, :, cut_i] = crop_norm_cutout(triplets[trip_i, :, :, cut_i],
                                                                     crop_to_size)

    return cropped_triplets


def make_triplet(alert, normalize: bool = True):
    """
    Unpack binary fits files containing cutouts from kowalski and preprocess
    images to mask NaNs and remove corrupted images
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
        3 channel 63 x 63 image representing the science, reference, and
        difference cutouts

    drop: bool
        whether or not the image is found to be corrupted

    Adapted from https://github.com/dmitryduev/braai
    """

    cutout_dict = dict()
    drop = False

    for cutout in ('science', 'template', 'difference'):
        cutout_data = loads(dumps([alert[f'cutout{cutout.capitalize()}']['stampData']]))[0]
        # unzip fits file
        with gzip.open(io.BytesIO(cutout_data), 'rb') as f:
            with fits.open(io.BytesIO(f.read())) as hdu:
                data = hdu[0].data

                # Compute median value to detect corrupted cutouts
                median = np.nanmedian(data.flatten())

                # check if image is corrupted
                if median == np.nan or median == -np.inf or median == np.inf:
                    print(
                        "    ",
                        alert['candidate']['candid'],
                        "bad median (nan or inf)"
                    )
                    drop = True

                # Fill in nans with 0
                cutout_dict[cutout] = np.nan_to_num(data)

                # normalize with L2 norm
                if normalize and not drop:
                    cutout_dict[cutout] /= np.linalg.norm(cutout_dict[cutout])

                # If image is all zeros, image is corrupted
                if np.all(cutout_dict[cutout].flatten() == 0):
                    print(
                        "    ",
                        alert['candidate']['candid'],
                        "zero image"
                    )
                    drop = True

        # pad to 63x63 if smaller
        shape = cutout_dict[cutout].shape
        if shape != (63, 63):
            print(
                "    ",
                alert['candidate']['candid'],
                f"{cutout} {shape}",
            )
            # Execute padding
            cutout_dict[cutout] = np.pad(cutout_dict[cutout],
                                         [(0, 63 - shape[0]),
                                          (0, 63 - shape[1])],
                                         mode='constant',
                                         constant_values=1e-9)

    triplet = np.zeros((63, 63, 3))
    triplet[:, :, 0] = cutout_dict['science']
    triplet[:, :, 1] = cutout_dict['template']
    triplet[:, :, 2] = cutout_dict['difference']

    # unzipped triplet and corrupted flag
    return triplet, drop


def extract_triplets(alerts):
    """
    Takes in list of alerts with key 'triplet', returns triplet separated from
    alert metadata

    Parameters
    ----------
    alerts: list of dicts
        ZTF alerts packets on which to separate metadata and image cutouts

    Returns
    -------
    alerts: list of dicts
        ZTF alerts packets without images

    triplets: array of floats (Nx63x63x3)
        triplets separated from alerts
    """
    triplets = np.empty((len(alerts), 63, 63, 3))
    for i, alert in enumerate(alerts):
        triplets[i] = alert['triplet']

        alert.pop('triplet')
        alert.pop('cutoutScience')
        alert.pop('cutoutTemplate')
        alert.pop('cutoutDifference')

    return alerts, triplets


# def rerun_braai(triplets):
#     """
#     Reruns latest version of braai, ZTF deep real bogus (Duev+2019), on all alerts

#     Parameters
#     ----------
#     triplets: array, Nx63x63x3 (N=number of alerts)
#         triplets of all alerts to be rerun

#     Returns
#     -------
#     new_drb: array of floats
#         new real/bogus scores for each alert
#     """
#     if sys.platform == "darwin":
#         # Disable GPUs if running on macOS
#         print("disabling GPUs")
#         tf.config.set_visible_devices([], 'GPU')

#     # Load model
#     tf.keras.backend.clear_session()
#     braai = tf.keras.models.load_model("misc/supporting_models/braai_d6_m9.h5")

#     # Run braai
#     new_drb = braai.predict(triplets)

#     return np.transpose(new_drb)[0]


def query_nondet(objid, first_alert_jd):
    """
    Query for last non-detection before first detection

    Parameters
    ----------
    objid: str
        ZTFID objectId of source in question

    first_alert_jd: float
        jd of source's first detection

    Returns
    -------
    last_nondet_jd: float
        jd of last non-detection before first detection
        NaN if no leading non-detection found

    last_nondet_diffmaglim: float
        limiting magnitude of last non-detection before first detection
        NaN if no leading non-detection found
    """
    if k is None:
        print("Kowalski credentials were not found. \
               They must be set as environment variables KOWALSKI_USER and \
               KOWALSKI_PASS. \nQuerying Kowalski will not be possible.")
        return np.nan, np.nan

    query = {
        "query_type": "find",
        "query": {
            "catalog": "ZTF_alerts_aux",
            "filter": {
                '_id': objid,
            },
            "projection": {
                "_id": 0,
                "prv_candidates.jd": 1,
                "prv_candidates.diffmaglim": 1,
                "prv_candidates.magpsf": 1,
                # "prv_candidates.fid": 1
            }
        }
    }

    r = k.query(query)

    nondet_lc = r['kowalski']['data']

    # Empty if the source has never had non-detections
    if len(nondet_lc) == 0:
        return np.nan, np.nan

    prv = pd.DataFrame(nondet_lc[0]['prv_candidates'])

    # if only non-detections found
    if 'magpsf' not in prv.columns:
        prv['magpsf'] = np.nan

    if 'jd' not in prv.columns:
        return np.nan, np.nan

    # non-detections before first detection
    leading_nondets = prv[np.isnan(prv['magpsf']) & (prv['jd'] < first_alert_jd)]

    # if no leading non-detections found
    if len(leading_nondets) == 0:
        return np.nan, np.nan

    # last non-detection before first detection
    last_nondet = leading_nondets.sort_values('jd', ascending=False).iloc[0]

    return last_nondet['jd'], last_nondet['diffmaglim']


def prep_alerts(alerts, label, new_drb):
    """
    Reorganizes dict structure, adds values for custom features, and reruns
    braai on provided alert packets

    Parameters
    ----------
    alerts: list of dicts
        alerts on which to make changes

    label: list of ints
        BTS/not-BTS labels to be added to alert packets

    new_drb: array of floats
        deep real bogus scores regenerated for each alert by latest version of
        braai (Duev+2019)
        see rerun_braai()

    Returns
    -------
    alert_df: DataFrame
        alerts represented as DataFrame with custom features, new drb scores,
        and labels added in
    """
    cand_class_data = [alert['candidate'] | alert['classifications'] for alert in alerts]

    alert_df = pd.DataFrame(cand_class_data)
    alert_df.insert(0, "objectId", [alert['objectId'] for alert in alerts])
#     df.insert(1, "candid", [alert['candid'] for alert in alerts])

    # label must be int equalling 0, 1 or a list of 1s and 0s
    if isinstance(label, (list, np.ndarray)):
        assert (len(label) == len(alerts))
        alert_df.insert(2, "label", label)
    elif isinstance(label, int):
        alert_df.insert(2, "label", np.full((len(alerts),), label, dtype=int))

    # Add new braai scores to alerts
    alert_df["new_drb"] = new_drb

    # Custom features to add to metadata
    alert_df["peakmag"] = None
    alert_df["maxmag"] = None

    alert_df["peakmag_so_far"] = None
    alert_df["maxmag_so_far"] = None

    alert_df["age"] = None
    alert_df["days_since_peak"] = None
    alert_df["days_to_peak"] = None

    # alert_df["last_nondet_jd"] = None
    # alert_df["last_nondet_diffmaglim"] = None
    # alert_df["first_det_dm"] = None
    # alert_df["first_det_dmdt"] = None

    alert_df["nnotdet"] = alert_df["ncovhist"] - alert_df["ndethist"]

    for objid in tqdm.tqdm(pd.unique(alert_df['objectId'])):
        obj_alerts = alert_df.loc[alert_df["objectId"] == objid].sort_values(by="jd")

        peakmag = np.min(obj_alerts["magpsf"])
        maxmag = np.max(obj_alerts["magpsf"])

        alert_df.loc[alert_df['objectId'] == objid, "peakmag"] = peakmag
        alert_df.loc[alert_df['objectId'] == objid, "maxmag"] = maxmag

        for i in range(len(obj_alerts)):
            idx_cur = obj_alerts.index[i]
            idx_so_far = obj_alerts.index[0:i + 1]

            jd_first_alert = np.min((alert_df.loc[idx_cur, "jdstarthist"],
                                     np.min(obj_alerts['jd'])))

            peakmag_so_far = np.min(obj_alerts.loc[idx_so_far, "magpsf"])
            maxmag_so_far = np.max(obj_alerts.loc[idx_so_far, "magpsf"])

            alert_df.loc[idx_cur, "peakmag_so_far"] = peakmag_so_far
            alert_df.loc[idx_cur, "maxmag_so_far"] = maxmag_so_far

            jd_peak_so_far = obj_alerts.loc[
                obj_alerts['magpsf'] == peakmag_so_far, "jd"
            ].to_numpy()[0]

            alert_df.loc[idx_cur, "age"] = alert_df.loc[idx_cur, "jd"] - jd_first_alert
            alert_df.loc[idx_cur, "days_since_peak"] = alert_df.loc[idx_cur, "jd"] - jd_peak_so_far
            alert_df.loc[idx_cur, "days_to_peak"] = jd_peak_so_far - jd_first_alert

        nondet_jd, nondet_diffmaglim = query_nondet(objid, np.min(obj_alerts['jd']))

        alert_df.loc[alert_df['objectId'] == objid, "last_nondet_jd"] = nondet_jd
        alert_df.loc[alert_df['objectId'] == objid, "last_nondet_diffmaglim"] = nondet_diffmaglim

        # first_det_mag = obj_alerts.sort_values("jd", ascending=True).iloc[0]['magpsf']

        # if np.isnan(nondet_jd):
        #     # In this case, no leading non-detection found, so make basic
        #     # assumption of rise-time.
        #     dm = first_det_mag - 20.5  # 20.5 is roughly ZTF 30 s exposure depth
        #     dt = 2  # BTS is 2-day cadence
        # else:
        #     dm = first_det_mag - nondet_diffmaglim
        #     dt = np.min(obj_alerts['jd']) - nondet_jd

        # alert_df.loc[alert_df['objectId'] == objid, "first_det_dm"] = dm
        # alert_df.loc[alert_df['objectId'] == objid, "first_det_dmdt"] = dm/dt

    print("Arranged candidate data, inserted labels and custom cols")
    return alert_df
