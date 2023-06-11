import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
import json, datetime, os, sys

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from matplotlib.colors import LogNorm
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from manage_data import create_subset, create_cuts_str

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
})
plt.rcParams['axes.linewidth'] = 1.5
random_state = 2

def run_val(output_dir):
    print(f"*** Running validation on {output_dir} ***")

    with open(output_dir+"report.json", 'r') as f:
        report = json.load(f)
    model_dir = output_dir + "model/"
    config = report['Train_config']

    val_cuts_str = create_cuts_str(0, 
                                   bool(config['val_sne_only']),
                                   bool(config['val_keep_near_threshold']), 
                                   bool(config['val_rise_only']))

    if not (os.path.exists(f"data/val_triplets_v5{val_cuts_str}.npy") and 
            os.path.exists(f"data/val_cand_v5{val_cuts_str}.csv")):
        
        create_subset("val", 0, config['val_sne_only'], 
                      config['val_keep_near_threshold'], config['val_rise_only'])
    else:
        print("Validation data already present")

    cand = pd.read_csv(f"data/val_cand_v5{val_cuts_str}.csv")

    print(f'num_notbts: {np.sum(cand.label == 0)}')
    print(f'num_bts: {np.sum(cand.label == 1)}')

    triplets = np.load(f"data/val_triplets_v5{val_cuts_str}.npy", mmap_mode='r')
    assert not np.any(np.isnan(triplets))

    tf.keras.backend.clear_session()

    if sys.platform == "darwin":
        # Disable GPUs if on darwin (macOS)
        tf.config.set_visible_devices([], 'GPU')

    model = tf.keras.models.load_model(model_dir)
    
    raw_preds = model.predict([triplets, cand.loc[:,config["metadata_cols"]]], batch_size=config['batch_size'], verbose=1)
    preds = np.rint(np.transpose(raw_preds))[0].astype(int)
    labels = cand["label"].to_numpy(dtype=int)

    cand["raw_preds"] = raw_preds
    cand["preds"] = preds

    results = preds == cand["label"].to_numpy()
    print(f"Overall validation accuracy {100*np.sum(results) / len(results):.2f}%")
    
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
    all_count, _ = np.histogram(cand['magpsf']                    , bins=bins)
    TP_count, _  = np.histogram(cand['magpsf'].to_numpy()[TP_idxs], bins=bins)
    FP_count, _  = np.histogram(cand['magpsf'].to_numpy()[FP_idxs], bins=bins)
    TN_count, _  = np.histogram(cand['magpsf'].to_numpy()[TN_idxs], bins=bins)
    FN_count, _  = np.histogram(cand['magpsf'].to_numpy()[FN_idxs], bins=bins)

    narrow_bins = np.arange(17,21.00,0.25)
    all_count_nb, _ = np.histogram(cand['magpsf']                    , bins=narrow_bins)
    TP_count_nb, _  = np.histogram(cand['magpsf'].to_numpy()[TP_idxs], bins=narrow_bins)
    FP_count_nb, _  = np.histogram(cand['magpsf'].to_numpy()[FP_idxs], bins=narrow_bins)
    TN_count_nb, _  = np.histogram(cand['magpsf'].to_numpy()[TN_idxs], bins=narrow_bins)
    FN_count_nb, _  = np.histogram(cand['magpsf'].to_numpy()[FN_idxs], bins=narrow_bins)

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

    fig = plt.figure(figsize=(20, 22), dpi=400)
    main_grid = gridspec.GridSpec(4, 3, wspace=0.3, hspace=0.3)

    plt.suptitle(model_dir[:-7], size=28, y=0.92)
    
    ax1 = plt.Subplot(fig, main_grid[0])
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

    ax7 = plt.Subplot(fig, main_grid[6])
    ax8 = plt.Subplot(fig, main_grid[7])
    ax9 = plt.Subplot(fig, main_grid[8])

    ax10 = plt.Subplot(fig, main_grid[9])
    ax11 = plt.Subplot(fig, main_grid[10])
    ax12 = plt.Subplot(fig, main_grid[11])

    def gt1(alerts):
        return np.sum(alerts['preds']) >= 1
        
    def gt2(alerts):
        return np.sum(alerts['preds']) >= 2

    def gt3(alerts):
        return np.sum(alerts['preds']) >= 3
    
    def g1l20(alerts):
        return (np.sum(alerts['preds']) >= 1) and (np.sum(alerts['raw_preds'] < 0.5) < 20)

    policy_names = ["gt1", "gt2", "gt3"]
    policies = [gt1, gt2, gt3]
    CP_axes = [ax7, ax8, ax9]
    ST_axes = [ax10, ax11, ax12]

    policy_performance = dict.fromkeys(policy_names)

    # Get label and peakmag for each source (by taking all unique objectIds)
    policy_cand = pd.DataFrame(columns=["objectId", "label", "peakmag"])
    # Iterate over all alerts in validation set
    for i in cand.index:
        # If this objectId hasn't been seen,
        if cand.iloc[i]["objectId"] not in policy_cand["objectId"].to_numpy():
            # Select this source's objectId, label, and magpsf
            policy_cand.loc[len(policy_cand)] = (cand.iloc[i]["objectId"], 
                                                cand.iloc[i]["label"], 
                                                cand.iloc[i]["peakmag"])

    # For each policy
    for name, func, cp_ax, st_ax in zip(policy_names, policies, CP_axes[0:len(policy_names)], ST_axes[0:len(policy_names)]):
        # Initialize new columns
        policy_cand[name+"_pred"] = 0

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

        # _, ax_temp = plt.subplots()
        TP_count_policy, _  = np.histogram(policy_cand.loc[TP_idxs_policy, "peakmag"], bins=bright_narrow_bins)
        FP_count_policy, _  = np.histogram(policy_cand.loc[FP_idxs_policy, "peakmag"], bins=bright_narrow_bins)
        # TN_count_policy, _  = np.histogram(policy_cand.loc[TN_idxs_policy, "peakmag"], bins=bright_narrow_bins)
        FN_count_policy, _  = np.histogram(policy_cand.loc[FN_idxs_policy, "peakmag"], bins=bright_narrow_bins)

        if all((len(TP_idxs_policy) > 0, len(TN_idxs_policy) > 0, len(FP_idxs_policy) > 0, len(FN_idxs_policy) > 0)):
            precision = TP_count_policy/(TP_count_policy + FP_count_policy)
            recall = TP_count_policy/(TP_count_policy + FN_count_policy)

            overall_precision = np.sum(TP_count_policy)/(np.sum(TP_count_policy) + np.sum(FP_count_policy))
            overall_recall = np.sum(TP_count_policy)/(np.sum(TP_count_policy) + np.sum(FN_count_policy))
            
            cp_ax.step(bright_narrow_bins, 100*np.append(recall[0], recall), color='#263D65', label='Completeness', linewidth=3)
            cp_ax.step(bright_narrow_bins, 100*np.append(precision[0], precision), color='#FE7F2D', label='Purity', linewidth=3)
            
            cp_ax.axhline(100*overall_precision, color='#FE7F2D', linewidth=2, linestyle='dashed')
            cp_ax.axhline(100*overall_recall, color='#263D65', linewidth=2, linestyle='dashed')

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
            st_ax.hist(policy_cand[name+"_del_st"], bins=50, histtype='step', edgecolor='#654690', linewidth=3, label=name)
        else:
            precision = recall = overall_precision = overall_recall = med_del_st = -999.0

        policy_performance[name] = {
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "precision": precision,
            "recall": recall,
            "peakmag_bins": bright_narrow_bins,
            "med_del_st": med_del_st,
        }

        cp_ax.text(x=17.8, y=76.25, s=f"{name}\n({100*overall_recall:.0f}%,{100*overall_precision:.0f}%)", fontsize=28, fontweight='bold', c='#654690')
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

    plt.savefig(output_dir+"/"+os.path.basename(os.path.normpath(output_dir))+val_cuts_str+".pdf", bbox_inches='tight')
    plt.close()

    return {
        "roc_auc": roc_auc, "bal_acc": bal_acc, "bts_acc": bts_acc, 
        "notbts_acc": notbts_acc, "alert_precision": alert_precision,
        "alert_recall": alert_recall, "policy_performance": policy_performance
    }


if __name__ == "__main__":
    run_val(sys.argv[1])

    # import glob

    # models = glob.glob("models/vgg6_metadata_1_1-v5-n60-bs16/*/")
    # print(models)
    # for model in models:
    #     run_val(model)