import sys, numpy as np, pandas as pd


def only_pd_gr(trips, cand):
    cand['isdiffpos'] = [True if isdiffpos == 't' else False for isdiffpos in cand['isdiffpos']]

    cand_pd_gr = cand[(cand['isdiffpos']) & ((cand['fid'] == 1) | (cand['fid'] == 2))]
    triplets_pd_gr = trips[(cand['isdiffpos']) & ((cand['fid'] == 1) | (cand['fid'] == 2))]

    return triplets_pd_gr, cand_pd_gr


def create_train_data(set_names, cuts, N_max=None, seed=2):
    # concat
    # optionally save to disk? with provided name
    
    triplets = np.empty((0,63,63,3))
    cand = pd.DataFrame()

    for set_name in set_names:
        print(f"Working on {set_name} data")
        # load set
        set_trips = np.load(f"data/base_data/{set_name}_triplets.npy", mmap_mode='r')
        set_cand = pd.read_csv(f"data/base_data/{set_name}_candidates.csv", index_col=False)
        print("  Read")
        
        # run other optional cuts (ex: take only positive differences in g or r band)
        set_trips, set_cand = cuts(set_trips, set_cand)
        print("  Ran cuts")
        set_cand.reset_index(inplace=True, drop=True)

        # thin to N_max
        # plt.figure()
        # _ = plt.hist(set_cand['objectId'].value_counts(), histtype='step', bins=50)
        # plt.tight_layout()
        # plt.show()
        print(f"  Initial median of {int(np.median(set_cand['objectId'].value_counts()))} detections per object")
        print(f"  {len(pd.unique(set_cand['objectId']))} sources initially in {set_name}")

        if N_max is not None:
            drops = np.empty((0,), dtype=int)
            for ID in set(set_cand['objectId']):
                reps = np.argwhere(np.asarray(set_cand['objectId']) == ID).flatten()
                if len(reps) >= N_max:
                    np.random.seed(seed)
                    drops = np.concatenate((drops, np.random.choice(reps, len(reps)-N_max, replace=False)))
            
            set_trips = np.delete(set_trips, drops, axis=0)
            set_cand = set_cand.drop(index=drops)
            set_cand.reset_index(inplace=True)
            print(f"  Dropped {len(drops)} {set_name} alerts down to {N_max} max per obj")
            print(f"  {len(pd.unique(set_cand['objectId']))} sources in {set_name}")
        
        # concat
        triplets = np.concatenate((triplets, set_trips))
        cand = pd.concat((cand, set_cand))
        print(f"  Merged {set_name}")

    # or return?
    np.save(f"data/triplets_v4{ f'_n{N_max}' if N_max is not None else '' }.npy", triplets)
    cand.to_csv(f"data/candidates_v4{ f'_n{N_max}' if N_max is not None else '' }.csv", index=False)
    print("Wrote merged triplets and candidate data")
    del triplets, cand


def create_validation_data(set_names, ztfids_val):
    triplets = np.empty((0,63,63,3))
    cand = pd.DataFrame()

    for set_name in set_names:
        print(f"Working on {set_name} data")

        set_trips = np.load(f"data/base_data/{set_name}_triplets.npy", mmap_mode='r')
        set_cand = pd.read_csv(f"data/base_data/{set_name}_candidates.csv", index_col=False)
        print("  Read")

        set_cand.reset_index(inplace=True, drop=True)
        set_trips, set_cand = val_helper(set_trips, set_cand, ztfids_val)
        print("  Ran cuts")

        if set_name in ["trues", "dims", "MS"]: 
            print("  thinning for", set_name)
            for obj_id in pd.unique(set_cand['objectId']):
                obj_mask = set_cand['objectId'] == obj_id
                obj_idx = set_cand.index[obj_mask]

                obj_cand = set_cand[obj_mask]

                if np.min(obj_cand['magpsf']) > 18.4 and np.min(obj_cand['magpsf']) < 18.6:
                    set_trips = np.delete(set_trips, obj_idx, axis=0)
                    set_cand = set_cand.drop(obj_idx)
                    set_cand.reset_index(inplace=True, drop=True)
        
        triplets = np.concatenate((triplets, set_trips))
        cand = pd.concat((cand, set_cand))
        cand.reset_index(inplace=True, drop=True)
        print(f"  Merged {set_name}")
        
    np.save("data/triplets_v4_val.npy", triplets)
    cand.reset_index(inplace=True, drop=True)
    cand.to_csv("data/candidates_v4_val.csv", index=False)


def val_helper(trips, cand, ztfids_val):
    trips_pd_gr, cand_pd_gr = only_pd_gr(trips, cand)

    is_val = cand_pd_gr['objectId'].isin(ztfids_val)
    cand_val = cand_pd_gr.loc[is_val]
    cand_val.reset_index(inplace=True, drop=True)
    trips_val = trips_pd_gr[is_val]
    
    return trips_val, cand_val

if __name__ == "__main__":
    create_train_data(["trues", "dims", "vars", "MS"], only_pd_gr, name="pd_gr", N_max=int(sys.argv[1]))