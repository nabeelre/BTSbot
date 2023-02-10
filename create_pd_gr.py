import sys, numpy as np, pandas as pd


def only_pd_gr(trips, cand):
    cand['isdiffpos'] = [True if isdiffpos == 't' else False for isdiffpos in cand['isdiffpos']]

    cand_pd_gr = cand[(cand['isdiffpos']) & ((cand['fid'] == 1) | (cand['fid'] == 2))]
    triplets_pd_gr = trips[(cand['isdiffpos']) & ((cand['fid'] == 1) | (cand['fid'] == 2))]

    return triplets_pd_gr, cand_pd_gr


def create_train_data(set_names, cuts, N_max=None, seed=2, sne_only=False):
    # concat
    # optionally save to disk? with provided name
    
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


    for set_name in set_names:
        print(f"Working on {set_name} data")
        # load set
        set_trips = np.load(f"data/base_data/{set_name}_triplets.npy", mmap_mode='r')
        set_cand = pd.read_csv(f"data/base_data/{set_name}_candidates.csv", index_col=False)
        print("  Read")

        print(f"  Initial median of {int(np.median(set_cand['objectId'].value_counts()))} detections per object")
        print(f"  {len(pd.unique(set_cand['objectId']))} sources initially in {set_name}")
        
        if sne_only and set_name == "dims":
            # froms dims, remove things classified with non-SN types - keep unclassifieds
            print("  Removing non-SNe from dims")
            dims = pd.read_csv(f"data/base_data/dims.csv")

            non_SN_types = ["AGN", "AGN?", "bogus", "bogus?", "duplicate", 
                            "nova", "rock", "star", "varstar", "QSO", "CV", 
                            "CLAGN", "Blazar"]

            objids_to_remove = dims.loc[dims['type'].isin(non_SN_types), "ZTFID"].to_numpy()

            idxs = set_cand.loc[set_cand["objectId"].isin(objids_to_remove)].index

            set_trips = np.delete(set_trips, idxs, axis=0)
            set_cand = set_cand.drop(index=idxs)
            set_cand.reset_index(inplace=True, drop=True)
            print(f"  {len(pd.unique(set_cand['objectId']))} sources remaining in {set_name}")

        # run other optional cuts (ex: take only positive differences in g or r band)
        set_trips, set_cand = cuts(set_trips, set_cand)
        set_cand.reset_index(inplace=True, drop=True)
        print("  Ran cuts")
        print(f"  {len(pd.unique(set_cand['objectId']))} sources remaining in {set_name}")

        # thin to N_max
        # plt.figure()
        # _ = plt.hist(set_cand['objectId'].value_counts(), histtype='step', bins=50)
        # plt.tight_layout()
        # plt.show()
        if N_max is not None:
            drops = np.empty((0,), dtype=int)
            for ID in set(set_cand['objectId']):
                reps = np.argwhere(np.asarray(set_cand['objectId']) == ID).flatten()
                if len(reps) >= N_max:
                    np.random.seed(seed)
                    drops = np.concatenate((drops, np.random.choice(reps, len(reps)-N_max, replace=False)))
            
            set_trips = np.delete(set_trips, drops, axis=0)
            set_cand = set_cand.drop(index=drops)
            set_cand.reset_index(inplace=True, drop=True)
            print(f"  Dropped {len(drops)} {set_name} alerts down to {N_max} max per obj")
            print(f"  {len(pd.unique(set_cand['objectId']))} final sources in {set_name}")
        
        # concat
        triplets = np.concatenate((triplets, set_trips))
        cand = pd.concat((cand, set_cand))
        print(f"  Merged {set_name}")

    np.save(f"data/triplets_v4{mods_str}.npy", triplets)
    cand.to_csv(f"data/candidates_v4{mods_str}.csv", index=False)
    print("Wrote merged triplets and candidate data")
    del triplets, cand


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

        # if rise_only:
        #     print("  Removing non-rise-time alerts")
        #     to_drop = []
        #     for objid in pd.unique(set_cand['objectId']):
        #         obj_alerts = set_cand.loc[set_cand['objectId'] == objid, ["magpsf", 'jd']]

        #         jd_peak = obj_alerts.iloc[np.argmin(obj_alerts['magpsf']), -1]

        #         postpeak = obj_alerts.loc[obj_alerts['jd'] > jd_peak].index
        #         to_drop = np.append(to_drop, postpeak)

        #     set_cand.drop(to_drop.astype(int), inplace=True)
        #     set_cand.reset_index(drop=True, inplace=True)
        #     set_trips = np.delete(triplets, to_drop.astype(int), axis=0)
        #     print(f"  {len(cand)} sources remaining in {set_name}")

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
    create_train_data(["trues", "dims", "vars", "MS"], only_pd_gr, name="pd_gr", N_max=int(sys.argv[1]))