import numpy as np
import pandas as pd
import os


def only_pd_gr(trips, cand):
    cand['isdiffpos'] = [True if isdiffpos == 't' else False for isdiffpos in cand['isdiffpos']]

    cand_pd_gr = cand[(cand['isdiffpos']) & ((cand['fid'] == 1) | (cand['fid'] == 2))]
    triplets_pd_gr = trips[(cand['isdiffpos']) & ((cand['fid'] == 1) | (cand['fid'] == 2))]

    return triplets_pd_gr, cand_pd_gr


def only_pd_gr_ps(trips, cand):
    cand['isdiffpos'] = [True if isdiffpos == 't' else False for isdiffpos in cand['isdiffpos']]

    cand_pd_gr_ps = cand[
        (cand['isdiffpos']) &
        ((cand['fid'] == 1) | (cand['fid'] == 2)) &
        ((cand['sgscore1'] >= 0) | (cand['sgscore2'] >= 0))
    ]

    triplets_pd_gr_ps = trips[
        (cand['isdiffpos']) &
        ((cand['fid'] == 1) | (cand['fid'] == 2)) &
        ((cand['sgscore1'] >= 0) | (cand['sgscore2'] >= 0))
    ]

    return triplets_pd_gr_ps, cand_pd_gr_ps


def create_cuts_str(N_max_p: int, N_max_n: int, sne_only: bool,
                    keep_near_threshold: bool, rise_only: bool):
    cuts_str = ""
    if N_max_p:
        if N_max_p == N_max_n:
            cuts_str += f"_N{N_max_p}"
        else:
            cuts_str += f"_Np{N_max_p}"
            if N_max_n:
                cuts_str += f"n{N_max_n}"
    if sne_only:
        cuts_str += "_sne"
    if not keep_near_threshold:
        cuts_str += "_nnt"
    if rise_only:
        cuts_str += "_rt"

    return cuts_str


def merge_sets_across_split(set_names, split_name, version_name, seed=2):
    """
    Load candidates and triplets from all sets (i.e. trues, dims, etc.) of a
    specific split (i.e. train, val, test) and merge into a single file. Shuffle
    the merged data and save to disk.
    """
    cand = pd.concat(
        [
            pd.read_csv(f"data/base_data/{set_name}_{split_name}_cand_{version_name}.csv",
                        index_col=False)
            for set_name in set_names
        ]
    )
    cand.reset_index(inplace=True, drop=True)

    triplets = np.concatenate(
        [
            np.load(f"data/base_data/{set_name}_{split_name}_triplets_{version_name}.npy",
                    mmap_mode='r')
            for set_name in set_names
        ],
        axis=0
    )
    print(f"  Merged {split_name}")

    shuffle_idx = np.random.choice(np.arange(len(cand)), size=(len(cand),), replace=False)
    np.save(f"data/{split_name}_triplets_{version_name}.npy", triplets[shuffle_idx])
    cand.loc[shuffle_idx].to_csv(f"data/{split_name}_cand_{version_name}.csv", index=False)
    print(f"Wrote merged and shuffled {split_name} triplets and candidate data")

    del triplets, cand


def cut_set_and_assign_splits(set_name, cuts, version_name, seed=2):
    """
    Load candidates and triplets from a specific set (i.e. trues, dims, etc.),
    apply cuts, and assign train/val/test splits. Save the triplets and candidates
    to disk.
    """
    print(f"Working on {set_name} data")
    # load set
    set_trips = np.load(f"data/base_data/{set_name}_triplets.npy", mmap_mode='r')
    set_cand = pd.read_csv(f"data/base_data/{set_name}_candidates.csv", index_col=False)
    print("  Read")

    print(f"  {len(pd.unique(set_cand['objectId']))} sources initially in {set_name}")
    print(f"  Median of {int(np.median(set_cand['objectId'].value_counts()))} dets per object")

    # run other optional cuts (ex: take only positive differences in g or r band)
    set_trips, set_cand = cuts(set_trips, set_cand)
    set_cand.reset_index(inplace=True, drop=True)
    print("  Ran cuts")
    print(f"  {len(pd.unique(set_cand['objectId']))} sources remaining in {set_name}")

    set_cand['source_set'] = set_name
    set_cand['N'] = None
    set_cand['split'] = None

    set_cand['is_SN'] = False
    set_cand['near_threshold'] = np.bitwise_and(set_cand['peakmag'] > 18.4,
                                                set_cand['peakmag'] < 18.6)
    set_cand['is_rise'] = False

    np.random.seed(seed)
    splits = np.random.choice(
        ["train", "val", "test"],
        size=(len(pd.unique(set_cand['objectId'])),),
        p=[0.81, 0.09, 0.10]
    )

    for i, objid in enumerate(pd.unique(set_cand['objectId'])):
        obj_cand = set_cand[set_cand['objectId'] == objid]

        # Label rise alerts
        jd_peak = obj_cand.iloc[np.argmin(obj_cand['magpsf']), obj_cand.columns.get_loc("jd")]

        set_cand.loc[np.bitwise_and(set_cand['objectId'] == objid,
                                    set_cand['jd'] <= jd_peak), "is_rise"] = True

        # Label alerts with N
        N_tot = len(obj_cand)
        np.random.seed(seed)
        N_labels = np.random.choice(np.arange(1, N_tot + 1), size=(N_tot,), replace=False)

        set_cand.loc[set_cand['objectId'] == objid, "N"] = N_labels

        # Train/Val/Test split
        set_cand.loc[set_cand['objectId'] == objid, "split"] = splits[i]

    # Label as SN or not
    if set_name in ["trues", "extIas"]:
        set_cand['is_SN'] = True

    if set_name == "dims":
        # froms dims, remove things classified with non-SN types - keep unclassifieds
        dims = pd.read_csv("data/base_data/dims.csv")

        non_SN_types = ["AGN", "AGN?", "bogus", "bogus?", "duplicate",
                        "nova", "rock", "star", "varstar", "QSO", "CV", "CV?",
                        "CLAGN", "Blazar"]

        SN_objectIds = dims.loc[~dims['type'].isin(non_SN_types), "ZTFID"].to_numpy()

        set_cand.loc[set_cand['objectId'].isin(SN_objectIds), 'is_SN'] = True

        # remove bright sources in dims (peaked < 18.5 only in partnership data)
        # this is a bandaid fix to label noise identified after revealing the test split
        only_dim = set_cand['peakmag'] > 18.5
        set_trips, set_cand = apply_cut(set_trips, set_cand, only_dim)

    is_train = set_cand[set_cand['split'] == "train"].index
    is_val = set_cand[set_cand['split'] == "val"].index
    is_test = set_cand[set_cand['split'] == "test"].index

    for split_name, cand, trips in zip(
        ["train", "val", "test"],
        [set_cand.loc[is_train], set_cand.loc[is_val], set_cand.loc[is_test]],
        [set_trips[is_train], set_trips[is_val], set_trips[is_test]]
    ):
        np.save(f"data/base_data/{set_name}_{split_name}_triplets_{version_name}.npy", trips)
        cand.to_csv(f"data/base_data/{set_name}_{split_name}_cand_{version_name}.csv",
                    index=False)
        print(f"Wrote {set_name} {split_name} triplets and candidate data")

    del set_trips, set_cand


def apply_cut(trips, cand, keep_idxs):
    trips = trips[keep_idxs]
    cand = cand.loc[keep_idxs]
    cand.reset_index(inplace=True, drop=True)

    return trips, cand


def create_subset(split_name, version_name, N_max_p: int, N_max_n: int = 0,
                  sne_only: bool = False, keep_near_threshold: bool = True,
                  rise_only: bool = False):

    split_trip_path = f"data/{split_name}_triplets_{version_name}.npy"
    split_cand_path = f"data/{split_name}_cand_{version_name}.csv"

    if not (os.path.exists(split_trip_path) and os.path.exists(split_cand_path)):
        print("Parent split files absent")
        return

    trips = np.load(split_trip_path, mmap_mode='r')
    cand = pd.read_csv(split_cand_path, index_col=False)

    print(f"Read {split_name}")
    if N_max_p and not N_max_n:
        N_max_n = N_max_p

    cuts_str = create_cuts_str(N_max_p, N_max_n, sne_only, keep_near_threshold, rise_only)

    if N_max_p:
        mask = np.zeros(len(cand))

        for objid in pd.unique(cand['objectId']):
            obj_alerts = cand.loc[cand['objectId'] == objid]

            source_set = obj_alerts.iloc[0, obj_alerts.columns.get_loc("source_set")]

            if split_name == "train":
                if source_set == "trues":
                    # For trues, take random N_max_p alerts (train only)
                    mask[obj_alerts.index] = obj_alerts["N"] <= N_max_p
                elif source_set in ["dims", "rejects"]:
                    # For dims & rejects, take random N_max_n alerts (train only)
                    mask[obj_alerts.index] = obj_alerts["N"] <= N_max_n
            elif source_set in ["trues", "dims", "rejects"]:
                # For val and test, take all alerts from trues, dims, and rejects
                mask[obj_alerts.index] = 1

            if source_set in ["vars", "junk"]:
                # source_set = "vars," take latest N_max_n alerts
                N_max_n_latest_alerts = obj_alerts.sort_values(by='jd').iloc[-N_max_n:]

                mask[N_max_n_latest_alerts.index] = 1

            # elif source_set == "extIas":
            #     p_obj_alerts = cand.loc[(cand['objectId'] == objid) & (cand['label'] == 1)]
            #     if split_name == "train":
            #         mask[p_obj_alerts.index] = p_obj_alerts["N"] <= N_max_p
            #     else:
            #         mask[p_obj_alerts.index] = 1

            #     n_obj_alerts = cand.loc[(cand['objectId'] == objid) & (cand['label'] == 0)]
            #     mask[n_obj_alerts.index] = n_obj_alerts["N"] <= N_max_n

        trips, cand = apply_cut(trips, cand, np.where(mask == 1)[0])

    if sne_only:
        trips, cand = apply_cut(trips, cand, cand[cand["is_SN"]].index)

    if not keep_near_threshold:
        trips, cand = apply_cut(trips, cand, cand[~cand["near_threshold"]].index)

    if rise_only:
        trips, cand = apply_cut(trips, cand, cand[cand["is_rise"]].index)

    print(f"Created a {cuts_str} subset of {split_name}")
    np.save(f"data/{split_name}_triplets_{version_name}{cuts_str}.npy", trips)
    cand.to_csv(f"data/{split_name}_cand_{version_name}{cuts_str}.csv", index=False)
    print(f"Wrote triplets and candidate data for {cuts_str} subset of {split_name}")


def subsample_data(split, version, perc_to_keep=10, random_seed=2):
    np.random.seed(random_seed)

    fraction_to_keep = perc_to_keep / 100
    triplets = np.load(f"data/{split}_triplets_{version}_N100.npy")
    cand = pd.read_csv(f"data/{split}_cand_{version}_N100.csv")

    objs = pd.unique(cand['objectId'])
    subsampled_objs = np.random.choice(objs, size=int(len(objs) * fraction_to_keep), replace=False)

    subsampled_cand = cand[cand['objectId'].isin(subsampled_objs)]
    subsampled_triplets = triplets[subsampled_cand.index]

    np.save(f"data/{split}_triplets_{version}s{perc_to_keep}_N100.npy", subsampled_triplets)
    subsampled_cand.to_csv(f"data/{split}_cand_{version}s{perc_to_keep}_N100.csv", index=False)


if __name__ == "__main__":
    version = "v11"

    # cut_set_and_assign_splits("trues", only_pd_gr_ps, version_name=version)
    # cut_set_and_assign_splits("dims", only_pd_gr_ps, version_name=version)
    # cut_set_and_assign_splits("vars", only_pd_gr_ps, version_name=version)
    # cut_set_and_assign_splits("rejects", only_pd_gr_ps, version_name=version)

    # merge_sets_across_split(
    #     set_names=["trues", "dims", "vars", "rejects"], split_name="train",
    #     version_name=version, seed=2
    # )
    # merge_sets_across_split(
    #     set_names=["trues", "dims", "vars", "rejects"], split_name="val",
    #     version_name=version, seed=2
    # )
    # merge_sets_across_split(
    #     set_names=["trues", "dims", "vars", "rejects"], split_name="test",
    #     version_name=version, seed=2
    # )

    # create_subset("train", version_name=version, N_max_p=100, N_max_n=100)
    # create_subset("val", version_name=version, N_max_p=100, N_max_n=100)
    # create_subset("test", version_name=version, N_max_p=100, N_max_n=100)

    subsample_data("train", "v11", perc_to_keep=10)
    subsample_data("val", "v11", perc_to_keep=10)
    subsample_data("test", "v11", perc_to_keep=10)

    subsample_data("train", "v11", perc_to_keep=50)
    subsample_data("val", "v11", perc_to_keep=50)
    subsample_data("test", "v11", perc_to_keep=50)

