import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
import json, datetime, os, sys

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from manage_data import create_subset, create_cuts_str

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
})
plt.rcParams['axes.linewidth'] = 1.5
random_state = 2

def run_val(output_dir, config_path):
    with open(config_path, 'r') as f:
        hparams = json.load(f)

    with open(output_dir+"report.json", 'r') as f:
        report = json.load(f)
    model_dir = output_dir + "model/"

    val_cuts = hparams['val_cuts']
    val_cuts_str = create_cuts_str(0, 
                                   bool(val_cuts['sne_only']),
                                   bool(val_cuts['keep_near_threshold']), 
                                   bool(val_cuts['rise_only']))

    if not (os.path.exists(f"data/val_triplets_v5{val_cuts_str}.npy") and 
            os.path.exists(f"data/val_cand_v5{val_cuts_str}.csv")):
        
        create_subset("val", hparams['N_max'], val_cuts['sne_only'], 
                      val_cuts['keep_near_threshold'], val_cuts['rise_only'])
    else:
        print("Validation data already present")

    cand = pd.read_csv(f"data/val_cand_v5{val_cuts_str}.csv")

    print(f'num_notbts: {np.sum(cand.label == 0)}')
    print(f'num_bts: {np.sum(cand.label == 1)}')

    triplets = np.load(f"data/val_triplets_v5{val_cuts_str}.npy", mmap_mode='r')
    assert not np.any(np.isnan(triplets))

    metadata_cols = hparams["metadata_cols"]
    metadata_cols.append("label")

    tf.keras.backend.clear_session()

    if bool(hparams['dont_use_GPU']):
        # DISABLE ALL GPUs
        tf.config.set_visible_devices([], 'GPU')

    model = tf.keras.models.load_model(model_dir)
    
    raw_preds = model.predict([triplets, cand.loc[:,metadata_cols[:-1]]], batch_size=hparams['batch_size'], verbose=1)
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

    fig = plt.figure(figsize=(15, 12), dpi=250)
    main_grid = gridspec.GridSpec(3, 3, wspace=0.3, hspace=0.3)

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

    ax4 = plt.Subplot(fig, main_grid[3])
    hist, xbins, ybins, im = ax4.hist2d(cand['magpsf'], raw_preds[:,0], norm=LogNorm(), bins=14, range=[[15, 22], [0, 1]])
    ax4.set_xlabel('magpsf', size=18)
    ax4.set_ylabel('Score', size=18)
    ax4.xaxis.set_major_locator(MultipleLocator(1))
    ax4.yaxis.set_major_locator(MultipleLocator(0.2))

    dividex4 = make_axes_locatable(ax4)
    cax4 = dividex4.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax4, orientation='vertical')
    fig.add_subplot(ax4)

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

    ax6.bar(bins[:-1], TP_count,                                    align='edge', width=bins[1]-bins[0], color=colors[0], label='TP')
    ax6.bar(bins[:-1], FP_count, bottom=TP_count,                   align='edge', width=bins[1]-bins[0], color=colors[1], label='FP')
    ax6.bar(bins[:-1], TN_count, bottom=TP_count+FP_count,          align='edge', width=bins[1]-bins[0], color=colors[2], label='TN')
    ax6.bar(bins[:-1], FN_count, bottom=TP_count+FP_count+TN_count, align='edge', width=bins[1]-bins[0], color=colors[3], label='FN')

    ax6.axvline(18.5, c='k', linewidth=2, linestyle='dashed', alpha=1, zorder=10)
    ax6.legend(ncol=2, frameon=False)

    ax6.xaxis.set_major_locator(MultipleLocator(1))
    ax6.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax6.tick_params(direction='out', axis='x', length=2, width=1, bottom=True, pad=10)
    ax6.tick_params(axis='y', left=False, right=False)

    ax6.set_xlabel("PSF Magnitude", size=18)
    ax6.set_ylabel("# of alerts", size=18)
    fig.add_subplot(ax6)

    # /===================================================================/
    # Per-alert precision and recall

    ax7 = plt.Subplot(fig, main_grid[6])

    if len(TP_idxs) >= 0 and len(TN_idxs) > 0:
        ax7.step(narrow_bins[:-1], TP_count_nb/(TP_count_nb + FP_count_nb), color='green', label='Precision')
        ax7.axhline(len(TP_idxs)/(len(TP_idxs)+len(FP_idxs)), color='darkgreen', label='Overall Precision', linestyle='dashed')

        ax7.step(narrow_bins[:-1], TP_count_nb/(TP_count_nb + FN_count_nb), color='red', label='Recall')
        ax7.axhline(len(TP_idxs)/(len(TP_idxs)+len(FN_idxs)), color='darkred', label='Overall Recall', linestyle='dashed')

    ax7.axvline(18.5, c='k', linewidth=2, linestyle='dashed', zorder=10)
    ax7.grid(True, linewidth=.3)
    ax7.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax7.xaxis.set_major_locator(MultipleLocator(0.5))

    ax7.set_xlim([17,20.5])
    ax7.set_ylim([0,1])
    # ax7.set_ylabel("Precision", size=20)
    ax7.set_xlabel("PSF Magnitude", size=18)
    ax7.set_ylabel("Fraction of alerts", size=18)
    ax7.legend(prop={'size': 10})

    ax7.tick_params(direction='in', axis='both', length=3, width=1.5, bottom=True, left=True, right=True, pad=10)
    ax7.tick_params(which='minor', direction='in', axis='y', length=1.5, width=1.5, bottom=True, left=True, right=True, pad=10)
    fig.add_subplot(ax7)

    # /===================================================================/
    # Per-object Precision and Recall

    ax8_grid = gridspec.GridSpecFromSubplotSpec(2, 1, 
                                                subplot_spec=main_grid[7], 
                                                wspace=0.1, hspace=0.1)
    ax9_grid = gridspec.GridSpecFromSubplotSpec(2, 1, 
                                                subplot_spec=main_grid[8], 
                                                wspace=0.1, hspace=0.1)

    ax8_top = plt.Subplot(fig, ax8_grid[0])
    ax8_bot = plt.Subplot(fig, ax8_grid[1])

    ax9_top = plt.Subplot(fig, ax9_grid[0])
    ax9_bot = plt.Subplot(fig, ax9_grid[1])

    def gt1(alerts):
        if np.sum(alerts['preds']) >= 1:
            return True
        return False
    
    def gt2(alerts):
        if np.sum(alerts['preds']) >= 2:
            return True
        return False

    def gt3(alerts):
        if np.sum(alerts['preds']) >= 3:
            return True
        return False
    
    def combi(alerts):
        if np.max(alerts['preds'] > 0.95):
            return True
        if np.sum(alerts['preds']) >= 5:
            return True
        return False

    metric_names = ["gt1", "gt2", "gt3", "combi"]
    metric_funcs = [gt1, gt2, gt3, combi]
    metric_linestyles = ["solid", "dashed"]
    axes = [ax8_top, ax8_bot, ax9_top, ax9_bot]

    # Get label and peakmag for each source (by taking all unique objectIds)
    metric_cand = pd.DataFrame(columns=["objectId", "label", "peakmag"])
    # Iterate over all alerts in validation set
    for i in cand.index:
        # If this objectId hasn't been seen,
        if cand.iloc[i]["objectId"] not in metric_cand["objectId"].to_numpy():
            # Select this source's objectId, label, and magpsf
            metric_cand.loc[len(metric_cand)] = (cand.iloc[i]["objectId"], 
                                                cand.iloc[i]["label"], 
                                                cand.iloc[i]["peakmag"])

    # For each metric
    for name, func, ax in zip(metric_names, metric_funcs, axes[0:len(metric_names)]):
        # Initialize new columns
        metric_cand[name+"_pred"] = 0

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
                metric_cand.loc[metric_cand['objectId'] == obj_id, name+"_pred"] = int(met_pred)
    #             cand.loc[idx_cur, name+"_pred"] = met_pred
                
        metric_labels = metric_cand["label"].to_numpy()
        metric_preds  = metric_cand[name+"_pred"].to_numpy()
        bright_narrow_bins = np.arange(17.00, 18.50+0.25, 0.25)

        TP_mask_met = np.bitwise_and(metric_labels, metric_preds)
        TN_mask_met = 1-(np.bitwise_or(metric_labels, metric_preds))
        FP_mask_met = np.bitwise_and(1-metric_labels, metric_preds)
        FN_mask_met = np.bitwise_and(metric_labels, 1-metric_preds)

        TP_idxs_met = [ii for ii, mi in enumerate(TP_mask_met) if mi == 1]
        TN_idxs_met = [ii for ii, mi in enumerate(TN_mask_met) if mi == 1]
        FP_idxs_met = [ii for ii, mi in enumerate(FP_mask_met) if mi == 1]
        FN_idxs_met = [ii for ii, mi in enumerate(FN_mask_met) if mi == 1]

        # _, ax_temp = plt.subplots()
        TP_count_met, _  = np.histogram(metric_cand.loc[TP_idxs_met, "peakmag"], bins=bright_narrow_bins)
        FP_count_met, _  = np.histogram(metric_cand.loc[FP_idxs_met, "peakmag"], bins=bright_narrow_bins)
        # TN_count_met, _  = np.histogram(metric_cand.loc[TN_idxs_met, "peakmag"], bins=bright_narrow_bins)
        FN_count_met, _  = np.histogram(metric_cand.loc[FN_idxs_met, "peakmag"], bins=bright_narrow_bins)

        if len(TP_idxs_met) >= 0 and len(TN_idxs_met) > 0:
            precision = TP_count_met/(TP_count_met + FP_count_met)
            recall = TP_count_met/(TP_count_met + FN_count_met)
            
            ax.step(bright_narrow_bins, np.append(precision[0], precision), color='green', label='Precision')
            ax.step(bright_narrow_bins, np.append(recall[0], recall), color='red', label='Recall')
            # ax.axhline(len(TP_idxs_met)/(len(TP_idxs_met) + len(FP_idxs_met)), color='green', label='Precision '+name)
            # ax.axhline(len(TP_idxs_met)/(len(TP_idxs_met) + len(FN_idxs_met)), color='red', label='Recall '+name)
        
        ax.text(x=18, y=0.8, s=name, fontsize=14, fontweight='bold')
        ax.axvline(18.5, c='k', linewidth=1, linestyle='dashed', alpha=0.5, zorder=10)
        ax.grid(True, linewidth=.3)

        ax.set_xlim([17.0,18.5])
        ax.set_ylim([0.75,1.005])
        
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        # ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.tick_params(direction='in', axis='both', length=3, width=1.5, bottom=True, left=True, right=True, pad=10)
        ax.tick_params(which='minor', direction='in', axis='y', length=1.5, width=1.5, bottom=True, left=True, right=True, pad=10)

    ax8_top.axes.get_xaxis().set_ticklabels([])
    ax9_top.axes.get_xaxis().set_ticklabels([])

    ax8_bot.set_xlabel("Peak Magnitude", size=18)
    ax9_bot.set_xlabel("Peak Magnitude", size=18)
    ax8_top.set_ylabel("Fraction of objects", size=18)
    ax8_top.yaxis.set_label_coords(-0.15,0)
    ax8_top.legend(prop={'size': 10}, frameon=False)

    fig.add_subplot(ax8_top)
    fig.add_subplot(ax8_bot)
    fig.add_subplot(ax9_top)
    fig.add_subplot(ax9_bot)

    # /===================================================================/

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8_top, ax8_bot, ax9_top, ax9_bot, cax4]:
        ax.tick_params(which='both', width=1.5)

    plt.savefig(output_dir+"/"+os.path.basename(os.path.normpath(output_dir))+".pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_val(sys.argv[1], "train_config.json")