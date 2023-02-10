import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
import json, datetime, os, sys

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from create_pd_gr import create_validation_data

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
})
plt.rcParams['axes.linewidth'] = 1.5

def run_val(output_dir, config_path):
    with open(config_path, 'r') as f:
        hparams = json.load(f)

    with open(output_dir+"report.json", 'r') as f:
        report = json.load(f)
    model_dir = output_dir + "model/"

    df = pd.read_csv(f"data/candidates_v4_n{hparams['N_max']}.csv")
    random_state = 2
    split = 0.1
    ztfids_seen, _ = train_test_split(pd.unique(df['objectId']), test_size=split, random_state=random_state)
    _, ztfids_val = train_test_split(ztfids_seen, test_size=split, random_state=random_state)

    if not (os.path.exists("data/triplets_v4_val.npy") and 
            os.path.exists("data/candidates_v4_val.csv")):
        
        create_validation_data(["trues", "dims", "vars", "MS"],  ztfids_val)
    else:
        print("Validation data already present")

    cand = pd.read_csv("data/candidates_v4_val.csv")

    print(f'num_notbts: {np.sum(cand.label == 0)}')
    print(f'num_bts: {np.sum(cand.label == 1)}')

    triplets = np.load("data/triplets_v4_val.npy", mmap_mode='r')
    assert not np.any(np.isnan(triplets))

    metadata_cols = hparams["metadata_cols"]
    metadata_cols.append("label")

    tf.keras.backend.clear_session()

    if bool(hparams['dont_use_GPU']):
        # DISABLE ALL GPUs
        tf.config.set_visible_devices([], 'GPU')

    model = tf.keras.models.load_model(model_dir)
    
    raw_preds = model.predict([triplets, cand.loc[:,metadata_cols[:-1]]], batch_size=16, verbose=1)
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
    all_count, _, _ = plt.hist(cand['magpsf']                    , bins=bins)
    TP_count, _, _  = plt.hist(cand['magpsf'].to_numpy()[TP_idxs], bins=bins)
    FP_count, _, _  = plt.hist(cand['magpsf'].to_numpy()[FP_idxs], bins=bins)
    TN_count, _, _  = plt.hist(cand['magpsf'].to_numpy()[TN_idxs], bins=bins)
    FN_count, _, _  = plt.hist(cand['magpsf'].to_numpy()[FN_idxs], bins=bins)
    plt.close()

    narrow_bins = np.arange(17,21.00,0.25)
    all_count_nb, _, _ = plt.hist(cand['magpsf']                    , bins=narrow_bins)
    TP_count_nb, _, _  = plt.hist(cand['magpsf'].to_numpy()[TP_idxs], bins=narrow_bins)
    FP_count_nb, _, _  = plt.hist(cand['magpsf'].to_numpy()[FP_idxs], bins=narrow_bins)
    TN_count_nb, _, _  = plt.hist(cand['magpsf'].to_numpy()[TN_idxs], bins=narrow_bins)
    FN_count_nb, _, _  = plt.hist(cand['magpsf'].to_numpy()[FN_idxs], bins=narrow_bins)
    plt.close()

    # TP_frac = TP_count / all_count
    # FP_frac = FP_count / all_count
    # TN_frac = TN_count / all_count
    # FN_frac = FN_count / all_count

    # perobj_acc = np.zeros(len(ztfids_val))

    # for i, ztfid in enumerate(ztfids_val):  
    #     obj_trips = triplets[cand['objectId']==ztfid]
    #     obj_label = labels[cand['objectId']==ztfid][0]
        
    #     if "metadata" in model_dir:
    #         obj_df = cand[cand['objectId']==ztfid][metadata_cols]
    #         obj_preds = np.array(np.rint(model.predict([obj_trips, obj_df.iloc[:,:-1]], batch_size=hparams["batch_size"], verbose=0).flatten()), dtype=int)
    #     else:
    #         obj_preds = np.array(np.rint(model.predict(obj_trips, batch_size=hparams["batch_size"], verbose=0).flatten()), dtype=int)
        
    #     perobj_acc[i] = np.sum(obj_preds==obj_label)/len(obj_trips)

    bts_acc = len(TP_idxs)/(len(TP_idxs)+len(FN_idxs))
    notbts_acc = len(TN_idxs)/(len(TN_idxs)+len(FP_idxs))
    bal_acc = (bts_acc + notbts_acc) / 2

    # /-----------------------------
    #  MAKE FIGURE
    # /-----------------------------
    # Accuracy

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(15, 12), dpi=250)

    plt.suptitle(model_dir[:-7], size=28)
    ax1.plot(report["Training history"]["accuracy"], label='Training', linewidth=2)
    ax1.plot(report['Training history']['val_accuracy'], label='Validation', linewidth=2)
    ax1.axhline(bts_acc, label="BTS", c='blue', linewidth=1.5, linestyle='dashed')
    ax1.axhline(notbts_acc, label="notBTS", c='green', linewidth=1.5, linestyle='dashed')
    ax1.axhline(bal_acc, label="Balanced", c='gray', linewidth=1.5, linestyle='dashed')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='best')
    # ax1.set_ylim([0.6,0.9])
    ax1.grid(True, linewidth=.3)

    # /===================================================================/
    # Loss

    ax2.plot(report['Training history']['loss'], label='Training', linewidth=2)
    ax2.plot(report['Training history']['val_loss'], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='best')
    # ax2.set_ylim([0.2,0.7])
    ax2.grid(True, linewidth=.3)

    # /===================================================================/
    # ROC

    ax3.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.plot(fpr, tpr, lw=2, label=f'ROC (area = {roc_auc:.5f})')
    ax3.set_xlabel('False Positive Rate (Contamination)')
    ax3.set_ylabel('True Positive Rate (Sensitivity)')
    ax3.legend(loc="lower right")
    ax3.grid(True, linewidth=.3)
    ax3.set(aspect='equal')

    # /===================================================================/
    # Per-alert precision and recall

    ax4.step(narrow_bins[:-1], TP_count_nb/(TP_count_nb + FP_count_nb), color='green', label='Precision')
    ax4.axhline(len(TP_idxs)/(len(TP_idxs)+len(FP_idxs)), color='darkgreen', label='Overall Precision', linestyle='dashed')

    ax4.step(narrow_bins[:-1], TP_count_nb/(TP_count_nb + FN_count_nb), color='red', label='Recall')
    ax4.axhline(len(TP_idxs)/(len(TP_idxs)+len(FN_idxs)), color='darkred', label='Overall Recall', linestyle='dashed')

    ax4.axvline(18.5, c='k', linewidth=2, linestyle='dashed', zorder=10)
    ax4.grid(True, linewidth=.3)
    ax4.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax4.xaxis.set_major_locator(MultipleLocator(0.5))

    ax4.set_xlim([17,20.5])
    ax4.set_ylim([0,1])
    # ax4.set_ylabel("Precision", size=20)
    ax4.set_xlabel("PSF Magnitude", size=18)
    ax4.set_ylabel("Fraction of alerts", size=18)
    ax4.legend(prop={'size': 10})

    ax4.tick_params(direction='in', axis='both', length=3, width=1.5, bottom=True, left=True, right=True, pad=10)
    ax4.tick_params(which='minor', direction='in', axis='y', length=1.5, width=1.5, bottom=True, left=True, right=True, pad=10)

    # /===================================================================/
    # Confusion Matrix

    ConfusionMatrixDisplay.from_predictions(labels, preds, normalize='true', 
                                            display_labels=["notBTS", "BTS"], 
                                            cmap=plt.cm.Blues, colorbar=False, ax=ax5)

    for im in plt.gca().get_images():
        im.set_clim(vmin=0,vmax=1)

    # /===================================================================/
    # Classification type hist vs mag

    colors = ['#26547C', '#A9BCD0', '#BA5A31', '#E59F71']

    ax6.bar(bins[:-1], TP_count,                                    align='edge', width=bins[1]-bins[0], color=colors[0], label='TP')
    ax6.bar(bins[:-1], FP_count, bottom=TP_count,                   align='edge', width=bins[1]-bins[0], color=colors[1], label='FP')
    ax6.bar(bins[:-1], TN_count, bottom=TP_count+FP_count,          align='edge', width=bins[1]-bins[0], color=colors[2], label='TN')
    ax6.bar(bins[:-1], FN_count, bottom=TP_count+FP_count+TN_count, align='edge', width=bins[1]-bins[0], color=colors[3], label='FN')

    ax6.axvline(18.5, c='k', linewidth=2, linestyle='dashed', alpha=1, zorder=10)
    ax6.legend(ncol=2, frameon=False)

    ax6.xaxis.set_major_locator(MultipleLocator(1))
    ax6.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax6.set_xlabel("PSF Magnitude", size=18)
    ax6.tick_params(direction='out', axis='x', length=2, width=1, bottom=True, pad=10)
    ax6.tick_params(axis='y', left=False, right=False)

    ax6.set_xlabel("PSF Magnitude", size=22)
    ax6.set_ylabel("# of alerts", size=22)

    # /===================================================================/
    # Per-object Precision and Recall

    def gt3(alerts):
        if np.sum(alerts['preds']) >= 3:
            return True
        return False

    metric_names = ["gt3"]
    metric_funcs = [gt3]
    metric_linestyles = ["solid"]

    # Get label and peakmag for each source (by taking all unique objectIds)
    metric_cand = pd.DataFrame(columns=["objectId", "label", "peakmag"])
    # Iterate over all alerts in validation set
    for i in cand.index:
        # If this objectId hasn't been seen,
        if cand.iloc[i]["objectId"] not in metric_cand["objectId"].to_numpy():
            # Select this source's objectId, label, and magpsf
            metric_cand.loc[len(metric_cand)] = (cand.iloc[i]["objectId"], 
                                                 cand.iloc[i]["label"], 
                                                 np.min(cand.loc[cand['objectId'] == cand.iloc[i]["objectId"], "magpsf"]))

    # For each metric
    for name, func, linestyle in zip(metric_names[0:1], metric_funcs[0:1], metric_linestyles[0:1]):
        # Initialize new columns
        metric_cand[[name+"_pred", name+"_select"]] = 0, 0

        # For each source
        for obj_id in metric_cand["objectId"]:
            # Pick out alerts for that source and sort them by time
            obj_alerts = cand.loc[cand["objectId"] == obj_id].sort_values(by="jd")

            # For each alert
            for i in range(len(obj_alerts)):
                # the obj_alerts index of the current row of iteration
                idx_cur = obj_alerts.index[i]

                # the obj_alerts index of the current and previous rows of iteration
                idx_sofar = obj_alerts.index[0:i+1]

                # Compute the prediction for this current metric
                met_pred = func(obj_alerts.loc[idx_sofar])
                
                # Store metric prediction and whether it was the first positive
                obj_alerts.loc[idx_cur, (name+"_pred", name+"_select")] = int(met_pred), int(met_pred and not np.any(obj_alerts.loc[idx_sofar, name+"_select"]))
 
        metric_labels = metric_cand["label"].to_numpy()
        metric_preds  = metric_cand[name+"_pred"].to_numpy()
        
        TP_mask_met = np.bitwise_and(metric_labels, metric_preds)
        TN_mask_met = 1-(np.bitwise_or(metric_labels, metric_preds))
        FP_mask_met = np.bitwise_and(1-metric_labels, metric_preds)
        FN_mask_met = np.bitwise_and(metric_labels, 1-metric_preds)

        TP_idxs_met = [ii for ii, mi in enumerate(TP_mask_met) if mi == 1]
        TN_idxs_met = [ii for ii, mi in enumerate(TN_mask_met) if mi == 1]
        FP_idxs_met = [ii for ii, mi in enumerate(FP_mask_met) if mi == 1]
        FN_idxs_met = [ii for ii, mi in enumerate(FN_mask_met) if mi == 1]

        TP_count_met, _, _  = ax8.hist(metric_cand.loc[TP_idxs_met, "peakmag"], bins=narrow_bins)
        FP_count_met, _, _  = ax8.hist(metric_cand.loc[FP_idxs_met, "peakmag"], bins=narrow_bins)
        TN_count_met, _, _  = ax8.hist(metric_cand.loc[TN_idxs_met, "peakmag"], bins=narrow_bins)
        FN_count_met, _, _  = ax8.hist(metric_cand.loc[FN_idxs_met, "peakmag"], bins=narrow_bins)
        ax8.clear()

        ax7.axhline(len(TP_idxs_met)/(len(TP_idxs_met) + len(FP_idxs_met)), color='green', label='Precision '+name)
        ax7.axhline(len(TP_idxs_met)/(len(TP_idxs_met) + len(FN_idxs_met)), color='red', label='Recall '+name)
        ax7.step(narrow_bins[:-1], TP_count_met/(TP_count_met + FP_count_met), color='green', label='Precision '+name, linestyle=linestyle)
        ax7.step(narrow_bins[:-1], TP_count_met/(TP_count_met + FN_count_met), color='red', label='Recall '+name, linestyle=linestyle)

    ax7.axvline(18.5, c='k', linewidth=1, linestyle='dashed', alpha=0.5, zorder=10)
    ax7.grid(True, linewidth=.3)
    ax7.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax7.xaxis.set_major_locator(MultipleLocator(0.5))

    ax7.set_xlim([17,20.5])
    ax7.set_ylim([0,1.01])
    ax7.set_ylabel("Precision", size=20)
    ax7.set_xlabel("PSF Magnitude", size=18)
    ax7.set_ylabel("Fraction of objects", size=18)
    ax7.legend(prop={'size': 10})

    ax7.tick_params(direction='in', axis='both', length=3, width=1.5, bottom=True, left=True, right=True, pad=10)
    ax7.tick_params(which='minor', direction='in', axis='y', length=1.5, width=1.5, bottom=True, left=True, right=True, pad=10)

    # /===================================================================/

    hist, xbins, ybins, im = ax8.hist2d(cand['distnr'], raw_preds[:,0], norm=LogNorm())
    ax8.set_xlabel('distnr')
    ax8.set_ylabel('Score')
    # ax8.xaxis.set_major_locator(MultipleLocator(1))
    ax8.yaxis.set_major_locator(MultipleLocator(0.2))

    divider8 = make_axes_locatable(ax8)
    cax8 = divider8.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax8, orientation='vertical')

    # /===================================================================/

    hist, xbins, ybins, im = ax9.hist2d(cand['magpsf'], raw_preds[:,0], norm=LogNorm(), bins=14, range=[[15, 22], [0, 1]])
    ax9.set_xlabel('magpsf')
    ax9.set_ylabel('Score')
    ax9.xaxis.set_major_locator(MultipleLocator(1))
    ax9.yaxis.set_major_locator(MultipleLocator(0.2))

    divider9 = make_axes_locatable(ax9)
    cax9 = divider9.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax9, orientation='vertical')

    # /===================================================================/

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, cax8, cax9]:
        ax.tick_params(which='both', width=1.5)

    plt.tight_layout()
    plt.savefig(output_dir+"/"+os.path.basename(os.path.normpath(output_dir))+".pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_val(sys.argv[1], "train_config.json")