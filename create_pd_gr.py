import sys, numpy as np, pandas as pd


def only_pd_gr(trips, cand):
    cand['isdiffpos'] = [True if isdiffpos == 't' else False for isdiffpos in cand['isdiffpos']]

    cand_pd_gr = cand[(cand['isdiffpos']) & ((cand['fid'] == 1) | (cand['fid'] == 2))]
    triplets_pd_gr = trips[(cand['isdiffpos']) & ((cand['fid'] == 1) | (cand['fid'] == 2))]

    return triplets_pd_gr, cand_pd_gr


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
        splits = np.random.choice(["train","val","test"], size=(len(pd.unique(set_cand['objectId'])),), p=[0.81,0.09,0.1])

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

    np.save(f"data/train_triplets_all.npy", train_triplets)
    train_cand.to_csv(f"data/train_cand_all.csv", index=False)
    print("Wrote merged train triplets and candidate data")

    np.save(f"data/val_triplets_all.npy", val_triplets)
    val_cand.to_csv(f"data/val_cand_all.csv", index=False)
    print("Wrote merged val triplets and candidate data")

    np.save(f"data/test_triplets_all.npy", test_triplets)
    test_cand.to_csv(f"data/test_cand_all.csv", index=False)
    print("Wrote merged test triplets and candidate data")

    del train_triplets, train_cand, val_triplets, val_cand, test_triplets, test_cand


def create_validation_data(set_names, ztfids_val, N_max=None, sne_only=False, keep_near_threshold=True, rise_only=False):
    triplets = np.empty((0,63,63,3))
    cand = pd.DataFrame()

    mods_str = ""
    if N_max is not None:
        mods_str += f"_n{N_max}"
    if sne_only:
        mods_str += "_sne"

        if "vars" in set_names:
            print("Vars should not be included in a SNe only compilation")
            print(set_names)
            exit()
    if not keep_near_threshold:
        mods_str += "_nnt"
    # if rise_only:
    #     mods_str += "_rt"


    for set_name in set_names:
        print(f"Working on {set_name} data")

        set_trips = np.load(f"data/base_data/{set_name}_triplets.npy", mmap_mode='r')
        set_cand = pd.read_csv(f"data/base_data/{set_name}_candidates.csv", index_col=False)
        print("  Read")

        # froms dims, remove things classified with non-SN types - keep unclassifieds
        if sne_only and set_name == "dims":
            dims = pd.read_csv(f"data/base_data/dims.csv")

            non_SN_types = ["AGN", "AGN?", "bogus", "bogus?", "duplicate", 
                            "nova", "rock", "star", "varstar", "QSO", "CV", 
                            "CLAGN", "Blazar"]

            objids_to_remove = dims.loc[dims['type'].isin(non_SN_types), "ZTFID"].to_numpy()

            idxs = set_cand.loc[set_cand["objectId"].isin(objids_to_remove)].index

            set_trips = np.delete(set_trips, idxs, axis=0)
            set_cand = set_cand.drop(index=idxs)
            set_cand.reset_index(inplace=True, drop=True)

        set_cand.reset_index(inplace=True, drop=True)
        set_trips, set_cand = val_helper(set_trips, set_cand, ztfids_val)
        print("  Ran cuts")

        if set_name in ["trues", "dims", "MS"]: 
            for obj_id in pd.unique(set_cand['objectId']):
                obj_mask = set_cand['objectId'] == obj_id
                obj_idx = set_cand.index[obj_mask]

                obj_cand = set_cand[obj_mask]

                if not keep_near_threshold:
                    print("  thinning near threshold", set_name)
                    if np.min(obj_cand['magpsf']) > 18.4 and np.min(obj_cand['magpsf']) < 18.6:
                        set_trips = np.delete(set_trips, obj_idx, axis=0)
                        set_cand = set_cand.drop(obj_idx)
                        set_cand.reset_index(inplace=True, drop=True)
        
        triplets = np.concatenate((triplets, set_trips))
        cand = pd.concat((cand, set_cand))
        cand.reset_index(inplace=True, drop=True)
        print(f"  Merged {set_name}")
        
    np.save(f"data/triplets_v4_val{mods_str}.npy", triplets)
    cand.reset_index(inplace=True, drop=True)
    cand.to_csv(f"data/candidates_v4_val{mods_str}.csv", index=False)


def val_helper(trips, cand, ztfids_val):
    trips_pd_gr, cand_pd_gr = only_pd_gr(trips, cand)

    is_val = cand_pd_gr['objectId'].isin(ztfids_val)
    cand_val = cand_pd_gr.loc[is_val]
    cand_val.reset_index(inplace=True, drop=True)
    trips_val = trips_pd_gr[is_val]
    
    return trips_val, cand_val

if __name__ == "__main__":
    merge_data(["trues", "dims", "vars", "MS"], only_pd_gr)