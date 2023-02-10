import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
import json, os, sys

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, mean_squared_error as mse
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

    if not (os.path.exists("data/triplets_v4_val_sne.npy") and 
            os.path.exists("data/candidates_v4_val_sne.csv")):

        df = pd.read_csv(f"data/candidates_v4_n{hparams['N_max']}_sne.csv")
        random_state = 2
        split = 0.1
        ztfids_seen, _ = train_test_split(pd.unique(df['objectId']), test_size=split, random_state=random_state)
        _, ztfids_val = train_test_split(ztfids_seen, test_size=split, random_state=random_state)
        
        create_validation_data(["trues", "dims", "MS"],  ztfids_val, sne_only=True)
    else:
        print("Validation data already present")

    cand = pd.read_csv("data/candidates_v4_val_sne.csv")
    triplets = np.load("data/triplets_v4_val_sne.npy", mmap_mode='r')
    assert not np.any(np.isnan(triplets))

    # Remove all alerts after peak
    print(len(cand), "total val alerts")
    to_drop = []
    for objid in pd.unique(cand['objectId']):
        obj_alerts = cand.loc[cand['objectId'] == objid, ["magpsf", 'jd']]

        jd_peak = obj_alerts.iloc[np.argmin(obj_alerts['magpsf']), -1]

        postpeak = obj_alerts.loc[obj_alerts['jd'] > jd_peak].index
        to_drop = np.append(to_drop, postpeak)

    cand.drop(to_drop.astype(int), inplace=True)
    cand.reset_index(drop=True, inplace=True)
    triplets = np.delete(triplets, to_drop.astype(int), axis=0)
    print(len(cand), "total val alerts during rise")

    metadata_cols = hparams["metadata_cols"]
    metadata_cols.append("peakmag")

    tf.keras.backend.clear_session()

    # if bool(hparams['dont_use_GPU']):
    #     # DISABLE ALL GPUs
    #     tf.config.set_visible_devices([], 'GPU')

    model = tf.keras.models.load_model(model_dir)
    
    peakmag_preds = model.predict([triplets, cand.loc[:,metadata_cols[:-1]]], batch_size=16, verbose=1)
    cand["peakmag_pred"] = peakmag_preds

    peakmags_labels = cand["peakmag"].to_numpy()
    cand["magpsf-peakmag"] = cand["magpsf"] - cand["peakmag"]

    val_loss = mse(peakmags_labels, peakmag_preds)
    print("Loss = " + str(val_loss))

    bins = np.arange(15,21.5,0.5)
    narrow_bins = np.arange(17,21.00,0.25)
    
    # TP_mask = np.bitwise_and(class_labels, class_preds)
    # TN_mask = 1-(np.bitwise_or(class_labels, class_preds))
    # FP_mask = np.bitwise_and(1-class_labels, class_preds)
    # FN_mask = np.bitwise_and(class_labels, 1-class_preds)

    # TP_idxs = [ii for ii, mi in enumerate(TP_mask) if mi == 1]
    # TN_idxs = [ii for ii, mi in enumerate(TN_mask) if mi == 1]
    # FP_idxs = [ii for ii, mi in enumerate(FP_mask) if mi == 1]
    # FN_idxs = [ii for ii, mi in enumerate(FN_mask) if mi == 1]

    # TP_count, _, _  = plt.hist(cand['magpsf'].to_numpy()[TP_idxs], bins=bins)
    # FP_count, _, _  = plt.hist(cand['magpsf'].to_numpy()[FP_idxs], bins=bins)
    # TN_count, _, _  = plt.hist(cand['magpsf'].to_numpy()[TN_idxs], bins=bins)
    # FN_count, _, _  = plt.hist(cand['magpsf'].to_numpy()[FN_idxs], bins=bins)
    # plt.close()

    # TP_count_nb, _, _  = plt.hist(cand['magpsf'].to_numpy()[TP_idxs], bins=narrow_bins)
    # FP_count_nb, _, _  = plt.hist(cand['magpsf'].to_numpy()[FP_idxs], bins=narrow_bins)
    # TN_count_nb, _, _  = plt.hist(cand['magpsf'].to_numpy()[TN_idxs], bins=narrow_bins)
    # FN_count_nb, _, _  = plt.hist(cand['magpsf'].to_numpy()[FN_idxs], bins=narrow_bins)
    # plt.close()

    # bts_acc = len(TP_idxs)/(len(TP_idxs)+len(FN_idxs))
    # notbts_acc = len(TN_idxs)/(len(TN_idxs)+len(FP_idxs))
    # bal_acc = (bts_acc + notbts_acc) / 2
    # print("BTS Accuracy", bts_acc)
    # print("not-BTS Accuracy", notbts_acc)
    # print("Balanced Accuracy", bal_acc)

    # /-----------------------------
    #  MAKE FIGURE
    # /-----------------------------
    # MSE

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(15, 12), dpi=250)

    plt.suptitle(model_dir[:-7], size=28)
    ax1.plot(report["Training history"]["loss"], label='Training', linewidth=2)
    ax1.plot(report['Training history']['val_loss'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE')
    ax1.set_ylim([0,10])
    ax1.legend(loc='best')
    ax1.grid(True, linewidth=.3)

    # /===================================================================/
    # RMSE and MSE vs magpsf
    rmses = np.zeros(len(bins)-1)
    mses = np.zeros(len(bins)-1)

    for i in range(len(bins[0:-1])):
        bin_cand = cand.loc[(cand['magpsf'] > bins[i]) & (cand['magpsf'] < bins[i+1]), ['peakmag_pred', 'peakmag']]

        rmses[i] = mse(bin_cand['peakmag'].to_numpy(), bin_cand['peakmag_pred'], squared=False)
        mses[i] = mse(bin_cand['peakmag'].to_numpy(), bin_cand['peakmag_pred'])

    ax2.step(bins, np.append(rmses[0], rmses), color='green', label='RMSE')
    ax2.step(bins, np.append(mses[0], mses), color='red', label='MSE')
    ax2.set_xlabel('magpsf')
    ax2.legend(loc='best')

    # /===================================================================/

    rmses = np.zeros(len(bins)-1)
    mses = np.zeros(len(bins)-1)

    for i in range(len(bins[0:-1])):
        bin_cand = cand.loc[(cand['peakmag'] > bins[i]) & (cand['peakmag'] < bins[i+1]), ['peakmag_pred', 'peakmag']]

        rmses[i] = mse(bin_cand['peakmag'].to_numpy(), bin_cand['peakmag_pred'], squared=False)
        mses[i] = mse(bin_cand['peakmag'].to_numpy(), bin_cand['peakmag_pred'])

    ax3.step(bins, np.append(rmses[0], rmses), color='green', label='RMSE')
    ax3.step(bins, np.append(mses[0], mses), color='red', label='MSE')
    ax3.set_xlabel('peakmag')
    ax3.legend(loc='best')

    # /===================================================================/

    hist, xbins, ybins, im = ax4.hist2d(cand['magpsf'], cand['peakmag_pred'], norm=LogNorm(), bins=32, range=[[15, 21], [15, 21]])
    ax4.plot([15,22], [15,22], c='k')
    ax4.set_xlabel('magpsf')
    ax4.set_ylabel('pred peakmag')
    ax4.xaxis.set_major_locator(MultipleLocator(1))
    ax4.yaxis.set_major_locator(MultipleLocator(1))

    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax4, orientation='vertical')

    # /===================================================================/

    hist, xbins, ybins, im = ax5.hist2d(cand['peakmag'], cand['peakmag_pred'], norm=LogNorm(), bins=32, range=[[15, 21], [15, 21]])
    ax5.plot([15,22], [15,22], c='k')
    ax5.set_xlabel('peakmag')
    ax5.set_ylabel('pred peakmag')
    ax5.xaxis.set_major_locator(MultipleLocator(1))
    ax5.yaxis.set_major_locator(MultipleLocator(1))

    divider5 = make_axes_locatable(ax5)
    cax5 = divider5.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax5, orientation='vertical')

    # /===================================================================/
    
    # 6

    # /===================================================================/
    
    bright_only = cand.loc[cand['peakmag'] < 18.5]

    abs_err = bright_only['peakmag_pred'] - bright_only['peakmag']

    ax7.scatter(bright_only['magpsf-peakmag'], abs_err, color='k', s=0.6, alpha=0.4)
    ax7.plot([-1,5], [-1,5], c='r', label="pred no change from current")
    ax7.plot([-1,5], [0,0], c='g', label="perfect pred")
    ax7.set_xlim([-0.03,2])
    ax7.set_ylim([-1,3])
    ax7.text(0.3, 1.5, "pred dimmer than current")
    ax7.text(1.3, 0.4, "pred brighter\nbut not enough")
    ax7.text(1 , -0.5, "pred too bright")
    ax7.set_xlabel('magpsf-peakmag')
    ax7.set_ylabel('peakmag_pred-peakmag')
    ax7.legend(loc='best')

    # /===================================================================/
    
    bright_and_valid = bright_only.loc[bright_only['peakmag_pred'] < bright_only['magpsf']]
    diff_bins = np.arange(0, 2.2, 0.2)

    rmses = np.zeros(len(diff_bins)-1)
    mses = np.zeros(len(diff_bins)-1)

    for i in range(len(diff_bins[0:-1])):
        bin_cand = bright_and_valid.loc[(bright_and_valid['magpsf-peakmag'] > diff_bins[i]) & (bright_and_valid['magpsf-peakmag'] < diff_bins[i+1]), ['peakmag_pred', 'peakmag', 'magpsf']]

        rmses[i] = mse(bin_cand['peakmag'].to_numpy(), bin_cand['peakmag_pred'], squared=False)
        mses[i] = mse(bin_cand['peakmag'].to_numpy(), bin_cand['peakmag_pred'])
        
    ax8.step(diff_bins, np.append(rmses[0], rmses), color='green', label='RMSE')
    ax8.step(diff_bins, np.append(mses[0], mses), color='red', label='MSE')
    ax8.set_xlabel('magpsf-peakmag')
    ax8.legend()

    # /===================================================================/

    hist, xbins, ybins, im = ax9.hist2d(bright_and_valid['magpsf'], bright_and_valid['peakmag_pred'], norm=LogNorm(), bins=32, range=[[15, 21], [15, 21]])
    ax9.plot([15,22], [15,22], c='k')
    ax9.set_xlim([15,20.5])
    ax9.set_ylim([15,20])
    ax9.set_xlabel('magpsf')
    ax9.set_ylabel('pred peakmag')
    ax9.xaxis.set_major_locator(MultipleLocator(1))
    ax9.yaxis.set_major_locator(MultipleLocator(1))

    divider9 = make_axes_locatable(ax9)
    cax9 = divider9.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax9, orientation='vertical')

    # /===================================================================/

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, cax4, cax5]:
        ax.tick_params(which='both', width=1.5)

    plt.tight_layout()
    plt.savefig(output_dir+"/"+os.path.basename(os.path.normpath(output_dir))+".pdf", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_val(sys.argv[1], "train_config_reg.json")