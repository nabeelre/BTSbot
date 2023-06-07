import os, numpy as np, pandas as pd


def only_pd_gr(trips, cand):
    cand['isdiffpos'] = [True if isdiffpos == 't' else False for isdiffpos in cand['isdiffpos']]

    cand_pd_gr = cand[(cand['isdiffpos']) & ((cand['fid'] == 1) | (cand['fid'] == 2))]
    triplets_pd_gr = trips[(cand['isdiffpos']) & ((cand['fid'] == 1) | (cand['fid'] == 2))]

    return triplets_pd_gr, cand_pd_gr


def create_cuts_str(N_max : int, sne_only : bool, 
                  keep_near_threshold : bool, rise_only : bool):
    cuts_str = ""
    if N_max:
        cuts_str += f"_n{N_max}"
    if sne_only:
        cuts_str += "_sne"
    if not keep_near_threshold:
        cuts_str += "_nnt"
    if rise_only:
        cuts_str += "_rt"

    return cuts_str


def merge_data(set_names, cuts, seed=2):

    train_triplets = np.empty((0,63,63,3))
    train_cand = pd.DataFrame()

    val_triplets = np.empty((0,63,63,3))
    val_cand = pd.DataFrame()

    test_triplets = np.empty((0,63,63,3))
    test_cand = pd.DataFrame()

    for set_name in set_names:
        print(f"Working on {set_name} data")
        # load set
        set_trips = np.load(f"data/base_data/{set_name}_triplets.npy", mmap_mode='r')
        set_cand = pd.read_csv(f"data/base_data/{set_name}_candidates.csv", index_col=False)
        print("  Read")

        print(f"  {len(pd.unique(set_cand['objectId']))} sources initially in {set_name}")
        print(f"  Median of {int(np.median(set_cand['objectId'].value_counts()))} detections per object")
        
        # run other optional cuts (ex: take only positive differences in g or r band)
        set_trips, set_cand = cuts(set_trips, set_cand)
        set_cand.reset_index(inplace=True, drop=True)
        print("  Ran cuts")
        print(f"  {len(pd.unique(set_cand['objectId']))} sources remaining in {set_name}")

        set_cand['source_set'] = set_name
        set_cand['N'] = None
        set_cand['split'] = None

        set_cand['is_SN'] = False
        set_cand['near_threshold'] = np.bitwise_and(set_cand['peakmag'] > 18.4, set_cand['peakmag'] < 18.6)
        set_cand['is_rise'] = False
        
        np.random.seed(seed)
        splits = np.random.choice(["train","val","test"], size=(len(pd.unique(set_cand['objectId'])),), p=[0.81,0.09,0.10])

        for i, objid in enumerate(pd.unique(set_cand['objectId'])):
            obj_cand = set_cand[set_cand['objectId'] == objid]
            
            # Label rise alerts
            jd_peak = obj_cand.iloc[np.argmin(obj_cand['magpsf']), obj_cand.columns.get_loc("jd")]
            
            set_cand.loc[np.bitwise_and(set_cand['objectId'] == objid, set_cand['jd'] <= jd_peak), "is_rise"] = True
            
            # Label alerts with N
            N_tot = len(obj_cand)
            np.random.seed(seed)
            N_labels = np.random.choice(np.arange(1,N_tot+1), size=(N_tot,), replace=False)
            
            set_cand.loc[set_cand['objectId'] == objid, "N"] = N_labels
            
            # Train/Val/Test split
            set_cand.loc[set_cand['objectId'] == objid, "split"] = splits[i]
            
        # Label as SN or not
        if set_name in ["trues", "MS"]:
            set_cand['is_SN'] = True
            
        if set_name == "dims":
            # froms dims, remove things classified with non-SN types - keep unclassifieds
            dims = pd.read_csv(f"data/base_data/dims.csv")

            non_SN_types = ["AGN", "AGN?", "bogus", "bogus?", "duplicate", 
                            "nova", "rock", "star", "varstar", "QSO", "CV", "CV?", 
                            "CLAGN", "Blazar"]

            SN_objectIds = dims.loc[~dims['type'].isin(non_SN_types), "ZTFID"].to_numpy()

            set_cand.loc[set_cand['objectId'].isin(SN_objectIds), 'is_SN'] = True
        
        is_train = set_cand[set_cand['split'] == "train"].index
        train_triplets = np.concatenate((train_triplets, set_trips[is_train]))
        train_cand = pd.concat((train_cand, set_cand.loc[is_train]))
        train_cand.reset_index(inplace=True, drop=True)

        is_val = set_cand[set_cand['split'] == "val"].index
        val_triplets = np.concatenate((val_triplets, set_trips[is_val]))
        val_cand = pd.concat((val_cand, set_cand.loc[is_val]))
        val_cand.reset_index(inplace=True, drop=True)

        is_test = set_cand[set_cand['split'] == "test"].index
        test_triplets = np.concatenate((test_triplets, set_trips[is_test]))
        test_cand = pd.concat((test_cand, set_cand.loc[is_test]))
        test_cand.reset_index(inplace=True, drop=True)
        print(f"  Merged {set_name}")

    for split_name, cand, trips in zip(["train", "val", "test"], 
                                       [train_cand, val_cand, test_cand], 
                                       [train_triplets, val_triplets, test_triplets]):
        np.random.seed(seed)
        shuffle_idx = np.random.choice(np.arange(len(cand)), size=(len(cand),), replace=False)
        np.save(f"data/{split_name}_triplets_v5.npy", trips[shuffle_idx])
        cand.loc[shuffle_idx].to_csv(f"data/{split_name}_cand_v5.csv", index=False)
        print(f"Wrote merged and shuffled {split_name} triplets and candidate data")

    del train_triplets, train_cand, val_triplets, val_cand, test_triplets, test_cand


def apply_cut(trips, cand, keep_idxs):
    trips = trips[keep_idxs]
    cand = cand.loc[keep_idxs]
    cand.reset_index(inplace=True, drop=True)
    
    return trips, cand


def create_subset(split_name, N_max : int = 0, sne_only : bool = False, 
                  keep_near_threshold : bool = True, rise_only : bool = False):

    split_trip_path = f"data/{split_name}_triplets_v5.npy"
    split_cand_path = f"data/{split_name}_cand_v5.csv"

    if not (os.path.exists(split_trip_path) and os.path.exists(split_cand_path)):
        print("Parent split files absent, creating them first")
        merge_data(["trues", "dims", "vars", "MS"], only_pd_gr)
    
    trips = np.load(split_trip_path, mmap_mode='r')
    cand = pd.read_csv(split_cand_path, index_col=False)
    
    print(f"Read {split_name}")
    cuts_str = create_cuts_str(N_max, sne_only, keep_near_threshold, rise_only)

    if N_max:
        mask = np.zeros(len(cand))

        for objid in pd.unique(cand['objectId']):
            obj_alerts = cand.loc[cand['objectId'] == objid]
            
            if obj_alerts.iloc[0, obj_alerts.columns.get_loc("source_set")] in ["trues", "dims", "MS"]:
                mask[obj_alerts.index] = obj_alerts["N"] <= N_max
            else:
                # source_set = "vars", take latest N_max alerts from vars sources
                N_max_latest_alerts = obj_alerts.sort_values(by='jd').iloc[-N_max:]
                
                mask[N_max_latest_alerts.index] = 1

        trips, cand = apply_cut(trips, cand, np.where(mask == 1)[0])

    if sne_only:
        trips, cand = apply_cut(trips, cand, cand[cand["is_SN"]].index)

    if not keep_near_threshold:
        trips, cand = apply_cut(trips, cand, cand[~cand["near_threshold"]].index)

    if rise_only:
        trips, cand = apply_cut(trips, cand, cand[cand["is_rise"]].index)

    print(f"Created a {cuts_str} subset of {split_name}")
    np.save(f"data/{split_name}_triplets_v5{cuts_str}.npy", trips)
    cand.to_csv(f"data/{split_name}_cand_v5{cuts_str}.csv", index=False)
    print(f"Wrote triplets and candidate data for {cuts_str} subset of {split_name}")


if __name__ == "__main__":
    # merge_data(["trues", "dims", "vars", "MS"], only_pd_gr)
    create_subset("train", 1)
    create_subset("train", 60)
