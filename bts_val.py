import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
import json, os, sys

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay

from matplotlib.colors import LogNorm
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from train_val_test_split import create_subset, create_cuts_str

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
})
plt.rcParams['axes.linewidth'] = 1.5
random_state = 2
run_test = False

def run_val(output_dir):
    if not run_test:
        split_name = "validation"
        split_name_short = "val"
    else:
        split_name = "test"
        split_name_short = "test"
    
    print(f"*** Running {split_name} on {output_dir} ***")

    if sys.platform == "darwin":
        # Disable GPUs if on darwin (macOS)
        print("disabling GPUs")
        tf.config.set_visible_devices([], 'GPU')

    try:
        with open(output_dir+"report.json", 'r') as f:
            report = json.load(f)
            config = report['Train_config']
    except:
        with open(output_dir+"train_config.json", 'r') as f:
            report = None
            config = json.load(f)

    if "N_maxs" in list(config):
        N_max_p = config["N_maxs"][0]
        N_max_n = config["N_maxs"][1]
    else:
        N_max_p = config["N_max_p"]
        if "N_max_n" in config:
            N_max_n = config["N_max_n"]
        else:
            N_max_n = N_max_p

    val_cuts_str = create_cuts_str(N_max_p, N_max_n,
                                   bool(config['val_sne_only']),
                                   bool(config['val_keep_near_threshold']), 
                                   bool(config['val_rise_only']))
    train_data_version = config['train_data_version']

    need_triplets = any([arch_type in output_dir for arch_type in ['mm_cnn', 'um_cnn']])
    need_metadata = any([arch_type in output_dir for arch_type in ['mm_cnn', 'um_nn']])

    triplets_present = os.path.exists(f"data/{split_name_short}_triplets_{train_data_version}{val_cuts_str}.npy")
    metadata_present = os.path.exists(f"data/{split_name_short}_cand_{train_data_version}{val_cuts_str}.csv")

    if (need_triplets and (not triplets_present)) or (not metadata_present):
        create_subset(split_name_short, train_data_version, N_max_p, N_max_n, 
                      config['val_sne_only'], config['val_keep_near_threshold'], 
                      config['val_rise_only'])
    else:
        print(f"{train_data_version}{val_cuts_str} {split_name} data already present")

    cand = pd.read_csv(f"data/{split_name_short}_cand_{train_data_version}{val_cuts_str}.csv")
    if need_triplets:
        triplets = np.load(f"data/{split_name_short}_triplets_{train_data_version}{val_cuts_str}.npy", mmap_mode='r')

    print(f'num_notbts: {np.sum(cand.label == 0)}')
    print(f'num_bts: {np.sum(cand.label == 1)}')

    if need_metadata:
        metadata_cols = config['metadata_cols']
        if cand[metadata_cols].isnull().values.any():
            print("Null in cand")
            exit(0)
    if need_triplets and np.any(np.isnan(triplets)):
        print("Null in triplets")
        exit(0)

    tf.keras.backend.clear_session()

    try:
        model = tf.keras.models.load_model(output_dir + "best_model/")
    except:
        print("couldn't find best_model/ trying to find model/")
        model = tf.keras.models.load_model(output_dir + "model/")
    
    if need_triplets and need_metadata:
        raw_preds = model.predict([triplets, cand.loc[:,metadata_cols]], batch_size=config['batch_size'], verbose=1)
    elif need_triplets:
        raw_preds = model.predict(triplets, batch_size=config['batch_size'], verbose=1)
    elif need_metadata:
        raw_preds = model.predict(cand.loc[:,metadata_cols], batch_size=config['batch_size'], verbose=1)
    
    preds = np.rint(np.transpose(raw_preds))[0].astype(int)
    labels = cand["label"].to_numpy(dtype=int)

    cand["raw_preds"] = raw_preds
    cand["preds"] = preds

    results = preds == cand["label"].to_numpy()
    print(f"Overall {split_name} accuracy {100*np.sum(results) / len(results):.2f}%")
    
    fpr, tpr, thresholds = roc_curve(labels, raw_preds)
    roc_auc = auc(fpr, tpr)
 
    TP_mask = np.bitwise_and(labels, preds)
    TN_mask = 1-(np.bitwise_or(labels, preds))
    FP_mask = np.bitwise_and(1-labels, preds)
    FN_mask = np.bitwise_and(labels, 1-preds)

    TP_idxs = [ii for ii, mi in enumerate(TP_mask) if mi == 1]
    TN_idxs = [ii for ii, mi in enumerate(TN_mask) if mi == 1]
    FP_idxs = [ii for ii, mi in enumerate(FP_mask) if mi == 1]
    FN_idxs = [ii for ii, mi in enumerate(FN_mask) if mi == 1]

    bins = np.arange(15,21.5,0.5)
    # all_count, _ = np.histogram(cand['magpsf']                    , bins=bins)
    TP_count, _  = np.histogram(cand['magpsf'].to_numpy()[TP_idxs], bins=bins)
    FP_count, _  = np.histogram(cand['magpsf'].to_numpy()[FP_idxs], bins=bins)
    TN_count, _  = np.histogram(cand['magpsf'].to_numpy()[TN_idxs], bins=bins)
    FN_count, _  = np.histogram(cand['magpsf'].to_numpy()[FN_idxs], bins=bins)

    # narrow_bins = np.arange(17,21.00,0.25)
    # all_count_nb, _ = np.histogram(cand['magpsf']                    , bins=narrow_bins)
    # TP_count_nb, _  = np.histogram(cand['magpsf'].to_numpy()[TP_idxs], bins=narrow_bins)
    # FP_count_nb, _  = np.histogram(cand['magpsf'].to_numpy()[FP_idxs], bins=narrow_bins)
    # TN_count_nb, _  = np.histogram(cand['magpsf'].to_numpy()[TN_idxs], bins=narrow_bins)
    # FN_count_nb, _  = np.histogram(cand['magpsf'].to_numpy()[FN_idxs], bins=narrow_bins)

    bts_acc = len(TP_idxs)/(len(TP_idxs)+len(FN_idxs))
    notbts_acc = len(TN_idxs)/(len(TN_idxs)+len(FP_idxs))
    bal_acc = (bts_acc + notbts_acc) / 2

    if len(TP_idxs) > 0 and len(TN_idxs) > 0:
        alert_precision = len(TP_idxs)/(len(TP_idxs)+len(FP_idxs))
        alert_recall = len(TP_idxs)/(len(TP_idxs)+len(FN_idxs))
    else:
        alert_precision = -999.0
        alert_recall = -999.0

    # /-----------------------------
    #  MAKE FIGURE
    # /-----------------------------
    print("Starting figure")
    # Accuracy

    fig = plt.figure(figsize=(20, 22), dpi=200)
    main_grid = gridspec.GridSpec(4, 3, wspace=0.3, hspace=0.3)

    plt.suptitle(output_dir, size=28, y=0.92)
    
    ax1 = plt.Subplot(fig, main_grid[0])
    if report:
        ax1.plot(report["Training history"]["accuracy"], label='Training', linewidth=2)
        ax1.plot(report['Training history']['val_accuracy'], label='Validation', linewidth=2)
    ax1.axhline(bts_acc, label="BTS", c='blue', linewidth=1.5, linestyle='dashed')
    ax1.axhline(notbts_acc, label="notBTS", c='green', linewidth=1.5, linestyle='dashed')
    ax1.axhline(bal_acc, label="Balanced", c='gray', linewidth=1.5, linestyle='dashed')
    ax1.set_xlabel('Epoch', size=18)
    ax1.set_ylabel('Accuracy', size=18)
    ax1.legend(loc='best')
    # ax1.set_ylim([0.6,0.9])
    ax1.grid(True, linewidth=.3)
    fig.add_subplot(ax1)

    # /===================================================================/
    # Loss

    ax2 = plt.Subplot(fig, main_grid[1])
    if report:
        ax2.plot(report['Training history']['loss'], label='Training', linewidth=2)
        ax2.plot(report['Training history']['val_loss'], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch', size=18)
    ax2.set_ylabel('Loss', size=18)
    ax2.legend(loc='best')
    # ax2.set_ylim([0.2,0.7])
    ax2.grid(True, linewidth=.3)
    fig.add_subplot(ax2)

    # /===================================================================/
    # ROC

    ax3 = plt.Subplot(fig, main_grid[2])
    ax3.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.plot(fpr, tpr, lw=2, label=f'ROC (area = {roc_auc:.5f})')
    ax3.set_xlabel('False Positive Rate (Contamination)')
    ax3.set_ylabel('True Positive Rate (Sensitivity)')
    ax3.legend(loc="lower right")
    ax3.grid(True, linewidth=.3)
    ax3.set(aspect='equal')
    fig.add_subplot(ax3)

    # /===================================================================/
    # 2d hist of score vs magpsf

    ax4_grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=main_grid[3], width_ratios=[1, 6], height_ratios=[1, 6], hspace=0.05, wspace=0.05)

    ax4 = fig.add_subplot(ax4_grid[1, 1])
    ax4_histx = fig.add_subplot(ax4_grid[0, 1], sharex=ax4)
    ax4_histy = fig.add_subplot(ax4_grid[1, 0], sharey=ax4)

    hist = ax4.hist2d(cand['magpsf'], raw_preds[:,0], norm=LogNorm(), bins=28, range=[[16, 21], [0, 1]], cmap=plt.cm.viridis)
    ax4.set_aspect('auto')

    ax4.xaxis.set_major_locator(mtick.MultipleLocator(1))
    ax4.xaxis.set_minor_locator(mtick.MultipleLocator(0.5))
    ax4.set_xlabel("PSF Magnitude", size=22)

    # main y axis
    ax4.yaxis.set_major_locator(mtick.MultipleLocator(0.2))
    ax4.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))

    # colorbar
    divider = make_axes_locatable(ax4)
    extend = divider.append_axes("right", "5%", pad=0.2)
    cb = plt.colorbar(hist[3], cax=extend)
    extend.set_ylabel('# of alerts', size=18)

    # mag marginal hist
    ax4_histx.hist(cand['magpsf'], bins=28, range=[16, 21], facecolor='#37125E')
    ax4_histx.tick_params(direction='out', axis='x', length=2, width=1, bottom=True, pad=10)
    ax4_histx.tick_params(axis='y', left=False, right=False)
    ax4_histx.set_yticklabels([])
    ax4_histx.set_aspect(0.00017, anchor='W')

    # score marginal hist
    ax4_histy.hist(raw_preds[:,0], orientation='horizontal', bins=28, range=[0, 1], facecolor='#37125E')
    ax4_histy.set_ylabel("Bright transient score", size=22)
    ax4_histy.tick_params(direction='out', axis='y', length=2, width=1, bottom=True, pad=10)
    ax4_histy.tick_params(axis='x', bottom=False)
    ax4_histy.set_xticklabels([])
    ax4_histy.invert_xaxis()
    ax4_histy.set_aspect('auto')

    for ax in [ax4, ax4_histx, ax4_histy]:
        try:
            ax.label_outer()
        except:
            pass

    fig.add_subplot(ax4)
    fig.add_subplot(ax4_histx)
    fig.add_subplot(ax4_histy)

    # /===================================================================/
    # Confusion Matrix

    ax5 = plt.Subplot(fig, main_grid[4])
    ConfusionMatrixDisplay.from_predictions(labels, preds, normalize='true', 
                                            display_labels=["notBTS", "BTS"], 
                                            cmap=plt.cm.Blues, colorbar=False, ax=ax5)

    for im in plt.gca().get_images():
        im.set_clim(vmin=0,vmax=1)
    fig.add_subplot(ax5)

    # /===================================================================/
    # Classification type hist vs mag

    ax6 = plt.Subplot(fig, main_grid[5])
    colors = ['#26547C', '#A9BCD0', '#BA5A31', '#E59F71']

    ax6.bar(bins[:-1], TP_count,                                    align='edge', width=bins[1]-bins[0], color=colors[0], label='TP', linewidth=0.1, edgecolor='k')
    ax6.bar(bins[:-1], FP_count, bottom=TP_count,                   align='edge', width=bins[1]-bins[0], color=colors[1], label='FP', linewidth=0.1, edgecolor='k')
    ax6.bar(bins[:-1], TN_count, bottom=TP_count+FP_count,          align='edge', width=bins[1]-bins[0], color=colors[2], label='TN', linewidth=0.1, edgecolor='k')
    ax6.bar(bins[:-1], FN_count, bottom=TP_count+FP_count+TN_count, align='edge', width=bins[1]-bins[0], color=colors[3], label='FN', linewidth=0.1, edgecolor='k')

    ax6.axvspan(10, 18.5, color='gold', alpha=0.2, lw=0)
    ax6.legend(ncol=2, frameon=False)

    ax6.set_xlim([16, 21])
    ax6.xaxis.set_major_locator(mtick.MultipleLocator(1))
    ax6.xaxis.set_minor_locator(mtick.MultipleLocator(0.5))
    ax6.tick_params(direction='out', axis='x', length=2, width=1, bottom=True, pad=10)
    ax6.tick_params(axis='y', left=False, right=False)

    ax6.set_xlabel("PSF Magnitude", size=18)
    ax6.set_ylabel("# of alerts", size=18)
    fig.add_subplot(ax6)

    # /===================================================================/
    # Per-object Precision and Recall

    save_times = pd.read_csv("data/base_data/trues.csv").set_index("ZTFID")['RCF_save_time'].to_dict()
    RCFJunk = pd.read_csv("data/base_data/RCFJunk_Jul25.csv", index_col=None)

    ax7 = plt.Subplot(fig, main_grid[6])
    ax8 = plt.Subplot(fig, main_grid[7])
    ax9 = plt.Subplot(fig, main_grid[8])

    ax10 = plt.Subplot(fig, main_grid[9])
    ax11 = plt.Subplot(fig, main_grid[10])
    ax12 = plt.Subplot(fig, main_grid[11])

    def bts_p1(alerts):
        valid = alerts[(alerts['preds'] == 1) & (alerts['magpsf'] < 19)]
        return len(valid) >= 2

    def bts_p2(alerts):
        if np.min(alerts['magpsf']) <= 18.5:
            valid = alerts[(alerts['preds'] == 1) & (alerts['magpsf'] < 19)]
            return len(valid) >= 2
        return False

    policy_names = ["bts_p1", "bts_p2"]
    policies = [bts_p1, bts_p2]
    CP_axes = [ax7, ax8, ax9, None, None]
    ST_axes = [ax10, ax11, ax12, None, None]

    policy_performance = dict.fromkeys(policy_names)

    # Get label and peakmag for each source (by taking all unique objectIds)
    policy_cand = pd.DataFrame(columns=["objectId", "label", "peakmag", 
                                        "remaining_alert_peakmag"])
    # Iterate over all alerts in validation/test set
    for objid in pd.unique(cand['objectId']):
        obj_alerts = cand[cand['objectId'] == objid]

        already_seen = objid in policy_cand['objectId'].to_numpy()
        in_RCFJunk = objid in RCFJunk['id'].to_numpy()
        good_coverage = len(obj_alerts) >= 2  # improve, change to quality cut?
        BTS_peak_thinned = (obj_alerts["label"].iloc[0] == 1) and np.min(obj_alerts["magpsf"]) > 18.5 

        if (not already_seen) and (not in_RCFJunk) and (good_coverage) and (not BTS_peak_thinned):
            policy_cand.loc[len(policy_cand)] = (objid,
                                                 cand.loc[cand['objectId'] == objid, "label"].iloc[0],
                                                 cand.loc[cand['objectId'] == objid, "peakmag"].iloc[0],
                                                 np.min(cand.loc[cand['objectId'] == objid, "magpsf"]))

    # For each policy
    for name, func, cp_ax, st_ax in zip(policy_names, policies, CP_axes, ST_axes):
        plot_policy = cp_ax != None
        # Initialize new columns
        policy_cand[name+"_pred"] = 0
        policy_cand[name+"_save_jd"] = -1
        policy_cand[name+"_save_mag"] = -1
        policy_cand[name+"_del_st"] = np.nan

        # For each source
        for obj_id in policy_cand["objectId"]:
            # Pick out alerts for that source and sort them by time
            obj_alerts = cand.loc[cand["objectId"] == obj_id].sort_values(by="jd")

            # For each alert
            for i in range(len(obj_alerts)):
                # the obj_alerts index of the current row of iteration
                idx_cur = obj_alerts.index[i]

                # the obj_alerts index of the current and previous rows of iteration
                idx_sofar = obj_alerts.index[0:i+1]

                # Don't save before 19 mag
                if np.min(obj_alerts.loc[obj_alerts.index[0:i+1], 'magpsf']) > 19:
                    continue

                # Compute the prediction for this current policy
                policy_pred = func(obj_alerts.loc[idx_sofar])

                # If this is the first positive pred
                if int(policy_pred) and not policy_cand.loc[policy_cand['objectId'] == obj_id, name+"_pred"].values[0]:
                    policy_cand.loc[policy_cand['objectId'] == obj_id, name+"_save_jd"] = obj_alerts.loc[idx_cur, "jd"]
                    policy_cand.loc[policy_cand['objectId'] == obj_id, name+"_save_mag"] = obj_alerts.loc[idx_cur, "magpsf"]

                # Store policy prediction
                policy_cand.loc[policy_cand['objectId'] == obj_id, name+"_pred"] = int(policy_pred)
                
        policy_labels = policy_cand["label"].to_numpy()
        policy_preds  = policy_cand[name+"_pred"].to_numpy()
        bright_narrow_bins = np.arange(17.00, 18.50+0.25, 0.25)

        TP_mask_policy = np.bitwise_and(policy_labels, policy_preds)
        TN_mask_policy = 1-(np.bitwise_or(policy_labels, policy_preds))
        FP_mask_policy = np.bitwise_and(1-policy_labels, policy_preds)
        FN_mask_policy = np.bitwise_and(policy_labels, 1-policy_preds)

        TP_idxs_policy = [ii for ii, mi in enumerate(TP_mask_policy) if mi == 1]
        TN_idxs_policy = [ii for ii, mi in enumerate(TN_mask_policy) if mi == 1]
        FP_idxs_policy = [ii for ii, mi in enumerate(FP_mask_policy) if mi == 1]
        FN_idxs_policy = [ii for ii, mi in enumerate(FN_mask_policy) if mi == 1]

        TP_count_policy_binned, _  = np.histogram(policy_cand.loc[TP_idxs_policy, "remaining_alert_peakmag"], 
                                                  bins=bright_narrow_bins)
        FP_count_policy_binned, _  = np.histogram(policy_cand.loc[FP_idxs_policy, "remaining_alert_peakmag"], 
                                                  bins=bright_narrow_bins)
        # TN_count_policy_binned, _  = np.histogram(policy_cand.loc[TN_idxs_policy, "remaining_alert_peakmag"], 
        #                                            bins=bright_narrow_bins)
        FN_count_policy_binned, _  = np.histogram(policy_cand.loc[FN_idxs_policy, "remaining_alert_peakmag"], 
                                                  bins=bright_narrow_bins)

        if all((len(TP_idxs_policy) > 0, len(TN_idxs_policy) > 0)):
            policy_precision = len(TP_idxs_policy) / (len(FP_idxs_policy) + len(TP_idxs_policy))
            policy_recall = len(TP_idxs_policy) / (len(FN_idxs_policy) + len(TP_idxs_policy))

            binned_precision = TP_count_policy_binned/(TP_count_policy_binned + FP_count_policy_binned)
            binned_recall = TP_count_policy_binned/(TP_count_policy_binned + FN_count_policy_binned)

            if plot_policy:
                cp_ax.step(bright_narrow_bins, 100*np.append(binned_recall[0], binned_recall), color='#263D65', label='Completeness', linewidth=3)
                cp_ax.step(bright_narrow_bins, 100*np.append(binned_precision[0], binned_precision), color='#FE7F2D', label='Purity', linewidth=3)
                
                cp_ax.axhline(100*policy_precision, color='#FE7F2D', linewidth=2, linestyle='dashed')
                cp_ax.axhline(100*policy_recall, color='#263D65', linewidth=2, linestyle='dashed')

            # policy cand has only val for all sets
            # save_times has train+val+test but only BTSSE trues
            #     MS trues have unreliable/unrealistic save times, so keep them excluded from this analysis
            #     pre ~2021 sources have unreliable/unrealistic fritz save times because they were scanned on GROWTH, so exclude them
            jan1_2021_jd = 2459215.5
            for objid in policy_cand.loc[TP_idxs_policy, "objectId"].to_list():
                if objid in list(save_times):
                    # Some BTS trues don't have save times...
                    if save_times[objid] >= jan1_2021_jd:
                        policy_cand.loc[policy_cand["objectId"] == objid, name+"_del_st"] = policy_cand.loc[policy_cand["objectId"] == objid, name+"_save_jd"].values[0] - save_times[objid]
        
            med_del_st = np.nanmedian(policy_cand[name+"_del_st"])
            if plot_policy:
                st_ax.hist(policy_cand[name+"_del_st"], bins=50, histtype='step', edgecolor='#654690', linewidth=3, label=name)
        else:
            policy_precision = policy_recall = binned_precision = binned_recall = med_del_st = -999.0

        policy_performance[name] = {
            "policy_precision": policy_precision,
            "policy_recall": policy_recall,
            "binned_precision": binned_precision,
            "binned_recall": binned_recall,
            "peakmag_bins": bright_narrow_bins,
            "med_del_st": med_del_st
        }

        if plot_policy:
            cp_ax.text(x=17.75, y=76.25, s=f"{name}\n({100*policy_recall:.0f}%,{100*policy_precision:.0f}%)", fontsize=20, fontweight='bold', c='#654690')
            cp_ax.axvline(18.5, c='k', linewidth=1, linestyle='dashed', alpha=0.5, zorder=10)
            cp_ax.grid(True, linewidth=.3)

            cp_ax.set_xlim([17.0,18.5])
            cp_ax.set_ylim([75,100.5])
            
            cp_ax.xaxis.set_major_locator(mtick.MultipleLocator(0.5))
            cp_ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.25))
            cp_ax.yaxis.set_major_locator(mtick.MultipleLocator(10))
            cp_ax.yaxis.set_minor_locator(mtick.MultipleLocator(5))

            cp_ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            
            cp_ax.tick_params(direction='in', axis='both', length=3, width=1.5, bottom=True, left=True, right=True, pad=10)
            cp_ax.tick_params(which='minor', direction='in', axis='y', length=1.5, width=1.5, bottom=True, left=True, right=True, pad=10)

            cp_ax.set_xlabel("Peak Magnitude", size=22)
            cp_ax.set_ylabel("% of objects", size=18)

            st_ax.axvline(med_del_st, linestyle='solid', c='k', linewidth=1.5, label=f"med:\n{med_del_st:.2f} d")
            st_ax.axvline(0, linestyle='dashed', c='gray', linewidth=1)

            st_ax.set_xlim([-15,15])
            st_ax.set_ylim([0,55])
            
            st_ax.legend(prop={'size': 20}, frameon=False)
            st_ax.set_ylabel("# of sources", size=18)
            st_ax.set_xlabel("Days after save by scanner", size=22)

        print(f"Finished policy {name} analysis")

    ax7.axhline(0, color='gray', linewidth=2, linestyle='dashed', label='Overall')
    ax7.legend(prop={'size': 14}, frameon=False, loc="lower left")

    fig.add_subplot(ax7)
    fig.add_subplot(ax8)
    fig.add_subplot(ax9)

    fig.add_subplot(ax10)
    fig.add_subplot(ax11)
    fig.add_subplot(ax12)

    # /===================================================================/

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, extend]:
        ax.tick_params(which='both', width=1.5)

    plt.savefig(f"{output_dir}/{os.path.basename(os.path.normpath(output_dir))}{val_cuts_str}{'_test' if run_test else ''}.pdf", bbox_inches="tight")
    

    print({
        "roc_auc": roc_auc, "bal_acc": bal_acc, "bts_acc": bts_acc, 
        "notbts_acc": notbts_acc, "alert_precision": alert_precision,
        "alert_recall": alert_recall, "policy_performance": policy_performance
    })

    return {
        "roc_auc": roc_auc, "bal_acc": bal_acc, "bts_acc": bts_acc, "fig": fig,
        "notbts_acc": notbts_acc, "alert_precision": alert_precision,
        "alert_recall": alert_recall, "policy_performance": policy_performance
    }


if __name__ == "__main__":
    run_val(sys.argv[1])
    