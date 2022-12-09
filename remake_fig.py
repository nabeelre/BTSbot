#!/usr/bin/env python3
import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json, os, sys
from astropy.stats import sigma_clipped_stats
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from pandas.plotting import register_matplotlib_converters, scatter_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
})
plt.rcParams['axes.linewidth'] = 1.5

def remake_fig(model_dir = None, N_max = None):
    batch_size = 64
    
    metadata_cols = ["sgscore1", "distpsnr1", "sgscore2", "distpsnr2", "sgscore3", "distpsnr3",
                     "fwhm", "magpsf", "sigmapsf", "ra", "dec", "diffmaglim", 
                     "classtar", "ndethist", "ncovhist", "sharpnr"]
    metadata_cols.append('label')

    if not model_dir:
        model_dir = sys.argv[1] #f"models/hd-vgg6-v4-n{N_max}/"
        metadata = "metadata" in sys.argv[1]
    else:
        metadata = "metadata" in model_dir
    # N_max = int(model_dir.rsplit("-n", 1)[1])

    print("Remaking figure for", model_dir)

    with open(model_dir+"/report.json", 'r') as f:
        report = json.load(f)
        
    df = pd.read_csv(f'data/candidates_v4_n{N_max}.csv')
    triplets = np.load(f'data/triplets_v4_n{N_max}.npy', mmap_mode='r')

    test_split = 0.1  # fraction of all data
    random_state = 2

    ztfids_seen, ztfids_test = train_test_split(pd.unique(df['objectId']), test_size=test_split, random_state=random_state)

    # Want array of indices for training alerts and testing alerts
    # Need to shuffle because validation is bottom 10% of train - shuffle test as well for consistency
    is_seen = df['objectId'].isin(ztfids_seen)
    is_test = ~is_seen
    mask_seen = shuffle(df.index.values[is_seen], random_state=random_state)
    mask_test  = shuffle(df.index.values[is_test], random_state=random_state)

    # x_seen, seen_df = triplets[mask_seen], df.loc[mask_seen][metadata_cols]
    # x_test, test_df = triplets[mask_test], df.loc[mask_test][metadata_cols]

    print(f"{len(ztfids_seen)} seen/train+val objects; {len(ztfids_test)} unseen/test objects")
    print(f"{100*(len(ztfids_seen)/len(pd.unique(df['objectId']))):.2f}%/{100*(len(ztfids_test)/len(pd.unique(df['objectId']))):.2f}% seen/unseen split by object\n")

    print(f"{len(mask_seen)} seen/train+val alerts; {len(mask_test)} unseen/test alerts")
    print(f"{100*(len(mask_seen)/len(df['objectId'])):.2f}%/{100*(len(mask_test)/len(df['objectId'])):.2f}% seen/unseen split by alert\n")


    validation_split = 0.1  # fraction of the seen data

    ztfids_train, ztfids_val = train_test_split(ztfids_seen, test_size=validation_split, random_state=random_state)

    is_train = df['objectId'].isin(ztfids_train)
    is_val = df['objectId'].isin(ztfids_val)
    mask_train = shuffle(df.index.values[is_train], random_state=random_state)
    mask_val  = shuffle(df.index.values[is_val], random_state=random_state)

    x_train, y_train = triplets[mask_train], df['label'][mask_train]
    x_val, y_val = triplets[mask_val], df['label'][mask_val]

    val_alerts = df.loc[mask_val]

    # train/val_df is a combination of the desired metadata and y_train/val (labels)
    # we provide the model a custom generator function to separate these as necessary
    train_df = df.loc[mask_train][metadata_cols]
    val_df   = val_alerts[metadata_cols]

    print(f"{len(ztfids_train)} train objects; {len(ztfids_val)} val objects")
    print(f"{100*(len(ztfids_train)/len(pd.unique(df['objectId']))):.2f}%/{100*(len(ztfids_val)/len(pd.unique(df['objectId']))):.2f}% train/val split by object\n")

    print(f"{len(x_train)} train alerts; {len(x_val)} val alerts")
    print(f"{100*(len(x_train)/len(df['objectId'])):.2f}%/{100*(len(x_val)/len(df['objectId'])):.2f}% train/val split by alert\n")

    model = tf.keras.models.load_model(model_dir+"/model/")
    preds = model.predict(x=x_val, batch_size=batch_size, verbose=1)
    labels_pred = np.rint(preds)

    fpr, tpr, thresholds = roc_curve(df['label'][mask_val], preds)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    _ = ConfusionMatrixDisplay.from_predictions(df.label.values[mask_val], 
                                            labels_pred, normalize='true', ax=ax)
    plt.close()

    # /-----------------------------
    #  OTHER FIGURE SET UP
    # /-----------------------------

    val_labels = np.array(list(map(int, df.label[mask_val]))).flatten()
    val_rpreds = np.array(list(map(int, np.rint(preds)))).flatten()

    val_TP_mask = np.bitwise_and(val_labels, val_rpreds)
    val_TN_mask = 1-(np.bitwise_or(val_labels, val_rpreds))
    val_FP_mask = np.bitwise_and(1-val_labels, val_rpreds)
    val_FN_mask = np.bitwise_and(val_labels, 1-val_rpreds)

    val_TP_idxs = [ii for ii, mi in enumerate(val_TP_mask) if mi == 1]
    val_TN_idxs = [ii for ii, mi in enumerate(val_TN_mask) if mi == 1]
    val_FP_idxs = [ii for ii, mi in enumerate(val_FP_mask) if mi == 1]
    val_FN_idxs = [ii for ii, mi in enumerate(val_FN_mask) if mi == 1]

    val_perobj_acc = np.zeros(len(ztfids_val))

    for i, ztfid in enumerate(ztfids_val):  
        trips = x_val[val_alerts['objectId']==ztfid]
        label = y_val[val_alerts['objectId']==ztfid].to_numpy()[0]
        
        if metadata:
            obj_df = val_alerts[val_alerts['objectId']==ztfid][metadata_cols]
            obj_preds = np.array(np.rint(model.predict([trips, obj_df.iloc[:,:-1]], batch_size=batch_size, verbose=0).flatten()), dtype=int)
        else:
            obj_preds = np.array(np.rint(model.predict(trips, batch_size=batch_size, verbose=0).flatten()), dtype=int)
        
        val_perobj_acc[i] = np.sum(obj_preds==label)/len(trips)

    train_loss = report['Training history']['loss']
    val_loss = report['Training history']['val_loss']

    train_acc = report['Training history']['accuracy']
    val_acc = report['Training history']['val_accuracy']

    bts_acc = len(val_TP_idxs)/(len(val_TP_idxs)+len(val_FN_idxs))
    notbts_acc = len(val_TN_idxs)/(len(val_TN_idxs)+len(val_FP_idxs))
    bal_acc = (bts_acc + notbts_acc) / 2

    # /-----------------------------
    #  MAKE FIGURE
    # /-----------------------------

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(15, 12), dpi=250)

    plt.suptitle(os.path.basename(os.path.normpath(model_dir)), size=28)
    ax1.plot(train_acc, label='Training', linewidth=2)
    ax1.plot(val_acc, label='Validation', linewidth=2)
    ax1.axhline(bts_acc, label="BTS", c='blue', linewidth=1.5, linestyle='dashed')
    ax1.axhline(notbts_acc, label="notBTS", c='green', linewidth=1.5, linestyle='dashed')
    ax1.axhline(bal_acc, label="Balanced", c='gray', linewidth=1.5, linestyle='dashed')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='best')
    # ax1.set_ylim([0.6,0.9])
    ax1.grid(True, linewidth=.3)

    # /===================================================================/

    ax2.plot(train_loss, label='Training', linewidth=2)
    ax2.plot(val_loss, label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='best')
    # ax2.set_ylim([0.2,0.7])
    ax2.grid(True, linewidth=.3)

    # /===================================================================/

    bins = np.arange(15,22,0.5)
    ax3.hist(df['magpsf'][mask_val].to_numpy()[val_TP_idxs], histtype='step', color='g', linewidth=2, label='TP', bins=bins, density=True, zorder=2)
    # ax3.hist(df['magpsf'][mask_val].to_numpy()[val_TN_idxs], histtype='step', color='b', linewidth=2, label='TN', bins=bins, density=True, zorder=3)
    ax3.hist(df['magpsf'][mask_val].to_numpy()[val_FP_idxs], histtype='step', color='r', linewidth=2, label='FP', bins=bins, density=True, zorder=4)
    # ax3.hist(df['magpsf'][mask_val].to_numpy()[val_FN_idxs], histtype='step', color='orange', linewidth=2, label='FN', bins=bins, density=True, zorder=5)
    ax3.axvline(18.5, c='k', linewidth=2, linestyle='dashed', label='18.5', alpha=0.5, zorder=10)
    ax3.legend(loc='upper left')
    ax3.set_xlabel('Magnitude')
    ax3.set_ylim([0,1])
    ax3.set_xlim([15,22])
    ax3.grid(True, linewidth=.3)
    ax3.xaxis.set_major_locator(MultipleLocator(1))
    ax3.yaxis.set_major_locator(MultipleLocator(0.2))

    # /===================================================================/

    # ax6.hist(df['magpsf'][mask_val].to_numpy()[val_TP_idxs], histtype='step', color='g', linewidth=2, label='TP', bins=bins, density=True, zorder=2)
    ax6.hist(df['magpsf'][mask_val].to_numpy()[val_TN_idxs], histtype='step', color='b', linewidth=2, label='TN', bins=bins, density=True, zorder=3)
    # ax6.hist(df['magpsf'][mask_val].to_numpy()[val_FP_idxs], histtype='step', color='r', linewidth=2, label='FP', bins=bins, density=True, zorder=4)
    ax6.hist(df['magpsf'][mask_val].to_numpy()[val_FN_idxs], histtype='step', color='orange', linewidth=2, label='FN', bins=bins, density=True, zorder=5)
    ax6.axvline(18.5, c='k', linewidth=2, linestyle='dashed', label='18.5', alpha=0.5, zorder=10)
    ax6.legend(loc='upper left')
    ax6.set_xlabel('Magnitude')
    ax6.set_ylim([0,1])
    ax6.set_xlim([15,22])
    ax6.grid(True, linewidth=.3)
    ax6.xaxis.set_major_locator(MultipleLocator(1))
    ax6.yaxis.set_major_locator(MultipleLocator(0.2))

    # /===================================================================/

    ax4.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.plot(fpr, tpr, lw=2, label=f'ROC (area = {roc_auc:.5f})')
    ax4.set_xlabel('False Positive Rate (Contamination)')
    ax4.set_ylabel('True Positive Rate (Sensitivity)')
    ax4.legend(loc="lower right")
    ax4.grid(True, linewidth=.3)
    ax4.set(aspect='equal')

    # /===================================================================/

    ConfusionMatrixDisplay.from_predictions(df.label.values[mask_val], 
                                            labels_pred, normalize='true', 
                                            display_labels=["notBTS", "BTS"], 
                                            cmap=plt.cm.Blues, colorbar=False, ax=ax5)

    # /===================================================================/

    # bins = np.arange(0,1.1,0.2)
    # ax7.hist(val_perobj_acc, histtype='step', color='k', linewidth=2, bins=bins, density=True)
    # ax7.set_xlabel('Accuracy')
    # ax7.xaxis.set_minor_locator(MultipleLocator(0.1))
    # ax7.grid(True, linewidth=.3)

    hist, xbins, ybins, im = ax7.hist2d(val_alerts['sgscore1'][val_alerts['sgscore1']>0], preds[:,0][val_alerts['sgscore1']>0], norm=LogNorm())
    ax7.set_xlabel('sgscore1')
    ax7.set_ylabel('Score')
    # ax7.xaxis.set_major_locator(MultipleLocator(1))
    ax7.yaxis.set_major_locator(MultipleLocator(0.2))

    divider7 = make_axes_locatable(ax7)
    cax7 = divider7.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax7, orientation='vertical')

    # /===================================================================/

    hist, xbins, ybins, im = ax8.hist2d(val_alerts['distnr'], preds[:,0], norm=LogNorm())
    ax8.set_xlabel('distnr')
    ax8.set_ylabel('Score')
    # ax8.xaxis.set_major_locator(MultipleLocator(1))
    ax8.yaxis.set_major_locator(MultipleLocator(0.2))

    divider8 = make_axes_locatable(ax8)
    cax8 = divider8.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax8, orientation='vertical')

    # /===================================================================/

    hist, xbins, ybins, im = ax9.hist2d(val_alerts['magpsf'], preds[:,0], norm=LogNorm(), bins=14, range=[[15, 22], [0, 1]])
    ax9.set_xlabel('magpsf')
    ax9.set_ylabel('Score')
    ax9.xaxis.set_major_locator(MultipleLocator(1))
    ax9.yaxis.set_major_locator(MultipleLocator(0.2))

    divider9 = make_axes_locatable(ax9)
    cax9 = divider9.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax9, orientation='vertical')

    # /===================================================================/

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, cax7, cax8, cax9]:
        ax.tick_params(which='both', width=1.5)

    plt.tight_layout()
    plt.savefig(model_dir+"/"+os.path.basename(os.path.normpath(model_dir))+"_new.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        import glob
        pres = glob.glob("models/*-n5-*")
        print(pres)
        for pre in pres:
            remake_fig(pre, 1)
    else:
        remake_fig(N_max=5)

