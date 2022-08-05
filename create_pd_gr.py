import sys, numpy as np, pandas as pd


def only_pd_gr(trips, cand):
    cand['isdiffpos'] = [True if isdiffpos == 't' else False for isdiffpos in cand['isdiffpos']]
    
    # Diagnostics
    # iband = cand[cand['fid']==3]
    # neg_iband = iband[~iband['isdiffpos']]

    # iband_objids = iband['objectId'].value_counts().index.to_numpy()
    # iband_counts = iband['objectId'].value_counts().to_numpy()

    # cand_objids = cand['objectId'].value_counts().index.to_numpy()
    # cand_counts = cand['objectId'].value_counts().to_numpy()

    # print(f"Percent of alerts that are in the i-band {100*len(iband)/len(cand):.2f}%")
    # print(f"Percent of objects that have at least one i band alert {100*len(iband_objids)/len(cand_objids):.2f}%")

    # neg_iband_objids = neg_iband['objectId'].value_counts().index.to_numpy()
    # neg_iband_counts = neg_iband['objectId'].value_counts().to_numpy()

    # print(f"Percent of i-band alerts that have negative differences {100*len(neg_iband)/len(iband):.2f}%")

    cand_pd_gr = cand[(cand['isdiffpos']) & ((cand['fid'] == 1) | (cand['fid'] == 2))]
    triplets_pd_gr = trips[(cand['isdiffpos']) & ((cand['fid'] == 1) | (cand['fid'] == 2))]

    # print("Positive difference alerts in g- or r-band", len(cand_pd_gr))
    return triplets_pd_gr, cand_pd_gr


def create_train_data(set_names, cuts, name, N_max=None, seed=2):
    np.random.seed(seed)
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
        print(f"  Initial median of {np.median(set_cand['objectId'].value_counts())} detections per object")
        
        if N_max is not None:
            drops = np.empty((0,), dtype=int)
            for ID in set(set_cand['objectId']):
                reps = np.argwhere(np.asarray(set_cand['objectId']) == ID).flatten()
                if len(reps) >= N_max:
                    drops = np.concatenate((drops, np.random.choice(reps, len(reps)-N_max, replace=False)))
            
            set_trips = np.delete(set_trips, drops, axis=0)
            set_cand = set_cand.drop(index=drops)
            set_cand.reset_index(inplace=True)
            print(f"  Dropped {len(drops)} {set_name} alerts down to {N_max} max per obj")
        
        # concat
        triplets = np.concatenate((triplets, set_trips))
        cand = pd.concat((cand, set_cand))
        print(f"  Merged {set_name}")

    # or return?
    np.save(f"data/triplets_{name}{ f'_{N_max}max' if N_max is not None else '' }.npy", triplets)
    cand.to_csv(f"data/candidates_{name}{ f'_{N_max}max' if N_max is not None else '' }.csv", index=False)
    print("Wrote merged triplets and candidate data")
    del triplets, cand

if __name__ == "__main__":
    create_train_data(['bts_true', 'bts_false', 'MS'], only_pd_gr, name="pd_gr", N_max=int(sys.argv[1]))