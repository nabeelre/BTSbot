import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
import json, datetime, os, sys

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.plotting import scatter_matrix

import CNN_models
from create_pd_gr import create_train_data, only_pd_gr

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
})
plt.rcParams['axes.linewidth'] = 1.5

# /-----------------------------
#  HYPERPARAMETERS
# /-----------------------------

loss = 'binary_crossentropy'
optimizer = tf.keras.optimizers.Adam(
    learning_rate=3e-4, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=None, 
    decay=0.0, 
    amsgrad=False
)
epochs = 500
patience = 150
weight_classes = True
batch_size = 64

# /-----------------------------
#  BASIC COMMAND LINE INTERFACE
# /-----------------------------

metadata = True if "meta" in sys.argv[1].lower() else False

if len(sys.argv) > 1:
    try: 
        model_type = getattr(CNN_models, sys.argv[1].lower())
    except:
        print("Could not find model of name", sys.argv[1], "defaulting to VGG6")
        model_type = CNN_models.vgg6
else:
    print("Defaulting to VGG6")
    model_type = CNN_models.vgg6

if len(sys.argv) > 2:
    try:
        N_max = int(sys.argv[2])
    except:
        print("Could not understand provided N_max=", sys.argv[2], "defaulting to N_max=10")
        N_max = 10
else:
    print("Defaulting to N_max=10")
    N_max = 10

metadata_cols = ['magpsf', 'distpsnr1', 'sgscore1', 'distpsnr2', 'distpsnr3', 'ra', 'dec', 'magnr', 'ndethist', 'neargaia', 'maggaia']
metadata_cols.append('label')

# /-----------------------------
#  LOAD TRAINING DATA
# /-----------------------------

if not (os.path.exists(f'data/candidates_v3_n{N_max}.csv') and 
        os.path.exists(f'data/triplets_v3_n{N_max}.npy')):
    create_train_data(['trues', 'dims', 'vars', 'MS'], only_pd_gr, name="pd_gr", N_max=N_max)
else:
    print("Training data already present")

df = pd.read_csv(f'data/candidates_v3_n{N_max}.csv')

print(f'num_notbts: {np.sum(df.label == 0)}')
print(f'num_bts: {np.sum(df.label == 1)}')

triplets = np.load(f'data/triplets_v3_n{N_max}.npy', mmap_mode='r')
assert not np.any(np.isnan(triplets))

# /-----------------------------
#  TRAIN/VAL/TEST SPLIT
# /-----------------------------

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

# /-----------------------------
#  SET UP CALLBACKS
# /-----------------------------

tf.keras.backend.clear_session()

# halt training if no gain in validation accuracy over patience epochs
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1, 
    patience=patience
)

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir="tb_logs/", 
#     histogram_freq=1,
#     write_graph=True,
#     write_images=True,
    update_freq='epoch',
#     profile_batch=1
)

# /-----------------------------
#  SET UP DATA GENERATORS WITH AUGMENTATION
# /-----------------------------

def rotate_incs_90(img):
    return np.rot90(img, np.random.choice([-1, 0, 1, 2]))

data_aug = {
    'h_flip': False,
    'v_flip': True,
    'fill_mode': 'constant',
    'cval': 0,
    'rot': True
}

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=data_aug['h_flip'],
    vertical_flip  =data_aug['v_flip'],
    fill_mode      =data_aug['fill_mode'],
    cval           =data_aug['cval'],
    preprocessing_function = rotate_incs_90 if data_aug['rot'] else None
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

if metadata:
    t_generator = train_datagen.flow(x_train, train_df, batch_size=batch_size, seed=2, shuffle=False)
    v_generator = val_datagen.flow(x_val, val_df, batch_size=batch_size, seed=2, shuffle=False)

    def multiinput_train_generator():
        # to keep track of complete epoch
        count = 0 
        while True:
            if count == len(train_df.index):
                # if the count is matching with the length of df, 
                # the one pass is completed, so reset the generator
                print("RESET HAPPENING")
                t_generator.reset()
                break
            count += 1
            # get the data from the generator
            # data is [[img], [other_cols]],
            data = t_generator.next()

            imgs = data[0]
            cols = data[1][:,:-1]
            targets = data[1][:,-1:]

            yield [imgs, cols], targets

    def multiinput_val_generator():
        # to keep track of complete epoch
        count = 0 
        while True:
            if count == len(val_df.index):
                # if the count is matching with the length of df, 
                # the one pass is completed, so reset the generator
                print("RESET HAPPENING")
                v_generator.reset()
                break
            count += 1
            # get the data from the generator
            # data is [[img], [other_cols]],
            data = v_generator.next()

            imgs = data[0]
            cols = data[1][:,:-1]
            targets = data[1][:,-1:]

            yield [imgs, cols], targets

    training_generator = multiinput_train_generator()
    validation_generator = multiinput_val_generator()
else:
    training_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, seed=2, shuffle=False)
    validation_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size, seed=2, shuffle=False)

# /-----------------------------
#  OTHER MODEL SET UP
# /-----------------------------

if weight_classes:
    # weight data class depending on number of examples?
    num_training_examples_per_class = np.array([len(y_train) - np.sum(y_train), np.sum(y_train)])
    assert 0 not in num_training_examples_per_class, 'found class without any examples!'

    # fewer examples -- larger weight
    weights = (1 / num_training_examples_per_class) / np.linalg.norm((1 / num_training_examples_per_class))
    normalized_weight = weights / np.max(weights)

    class_weight = {i: w for i, w in enumerate(normalized_weight)}
else:
    class_weight = {i: 1 for i in range(2)}
    
# image shape:
image_shape = x_train.shape[1:]
print('Input image shape:', image_shape)

# metadata shape (if necessary):
if metadata:
    metadata_shape = np.shape(train_df.iloc[0][:-1])
    print('Input metadata shape:', metadata_shape)
    model = model_type(image_shape=image_shape, metadata_shape=metadata_shape)
else:
    model = model_type(image_shape=image_shape)


run_t_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f'{model.name}-v3-n{N_max}'

# /-----------------------------
#  COMPILE AND TRAIN MODEL
# /-----------------------------

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

h = model.fit(
    training_generator,
    steps_per_epoch=0.8*len(x_train) // batch_size,
    validation_data=validation_generator,
    validation_steps=(0.8*len(x_val)) // batch_size,
    class_weight=class_weight,
    epochs=epochs,
    verbose=1, callbacks=[early_stopping, tensorboard]
)

# /-----------------------------
#  LOG MISCLASSIFICATIONS
# /-----------------------------

print('Evaluating on training set to check misclassified samples:')
if metadata:
    labels_training_pred = model.predict([x_train, train_df.iloc[:,:-1]], batch_size=batch_size, verbose=1)
else:
    labels_training_pred = model.predict(x_train, batch_size=batch_size, verbose=1)

# XOR will show misclassified samples
misclassified_train_mask = np.array(list(map(int, df.label[mask_train]))).flatten() ^ \
                           np.array(list(map(int, np.rint(labels_training_pred)))).flatten()

misclassified_train_mask = [ii for ii, mi in enumerate(misclassified_train_mask) if mi == 1]

misclassifications_train = {int(c): [int(l), float(p)]
                            for c, l, p in zip(df.candid.values[mask_train][misclassified_train_mask],
                                               df.label.values[mask_train][misclassified_train_mask],
                                               labels_training_pred[misclassified_train_mask])}

print('Evaluating on validation set for loss and accuracy:')
if metadata:
    preds = model.evaluate([x_val, val_df.iloc[:,:-1]], y_val, batch_size=batch_size, verbose=1)
else:
    preds = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=1)
val_loss = float(preds[0])
val_accuracy = float(preds[1])
print("Loss = " + str(val_loss))
print("Val Accuracy = " + str(val_accuracy))

print('Evaluating on validation set to check misclassified samples:')
if metadata:
    preds = model.predict(x=[x_val, val_df.iloc[:,:-1]], batch_size=batch_size, verbose=1)
else:
    preds = model.predict(x=x_val, batch_size=batch_size, verbose=1)
# XOR will show misclassified samples
misclassified_val_mask = np.array(list(map(int, df.label[mask_val]))).flatten() ^ \
                          np.array(list(map(int, np.rint(preds)))).flatten()
misclassified_val_mask = [ii for ii, mi in enumerate(misclassified_val_mask) if mi == 1]

misclassifications_val = {int(c): [int(l), float(p)]
                           for c, l, p in zip(df.candid.values[mask_val][misclassified_val_mask],
                                              df.label.values[mask_val][misclassified_val_mask],
                                              preds[misclassified_val_mask])}

# round probs to nearest int (0 or 1)
labels_pred = np.rint(preds)

# /-----------------------------
#  ROC CURVE AND CONFUSION MATRIX
# /-----------------------------

fpr, tpr, thresholds = roc_curve(df['label'][mask_val], preds)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
CFD = ConfusionMatrixDisplay.from_predictions(df.label.values[mask_val], 
                                        labels_pred, normalize='true', ax=ax)
plt.close()

# /-----------------------------
#  SAVE REPORT AND MODEL TO DISK
# /-----------------------------

# generate training report in json format
print('Generating report...')
report = {'Run time stamp': run_t_stamp,
     'Model name': model_name,
     'Model trained': model_type.__name__,
     'Batch size': batch_size,
     'Optimizer': str(type(optimizer)),
     'Requested number of train epochs': epochs,
     'Early stopping after epochs': patience,
     'Training+validation/test split': test_split,
     'Training/validation split': validation_split,
     'Weight training data by class': class_weight,
     'Random state': random_state,
     'Number of training examples': x_train.shape[0],
     'Number of val examples': x_val.shape[0],
     'X_train shape': x_train.shape,
     'Y_train shape': y_train.shape,
     'X_val shape': x_val.shape,
     'Y_val shape': y_val.shape,
     'Data augmentation': data_aug,
     'Confusion matrix': CFD.confusion_matrix.tolist(),
     'Misclassified val candids': list(misclassifications_val.keys()),
     'Misclassified training candids': list(misclassifications_train.keys()),
     'Val misclassifications': misclassifications_val,
     'Training misclassifications': misclassifications_train,
     'Training history': h.history
     }
for k in report['Training history'].keys():
    report['Training history'][k] = np.array(report['Training history'][k]).tolist()

report_dir = "models/"+model_name+"/"
model_dir = report_dir+"model/"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

f_name = os.path.join(report_dir, f'report.json')
with open(f_name, 'w') as f:
    json.dump(report, f, indent=2)

model.save(model_dir)
tf.keras.utils.plot_model(model, report_dir+"model_architecture.pdf", show_shapes=True, show_layer_names=False, show_layer_activations=True)

# with open(report_dir+'model_summary.txt', 'w') as f:
#     model.summary(print_fn=lambda x: f.write(x + '\n'), expand_nested=True, show_trainable=True)

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

# /-----------------------------
#  MAKE FIGURE
# /-----------------------------

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(15, 12), dpi=250)

plt.suptitle(os.path.basename(os.path.normpath(report_dir)), size=28)
ax1.plot(train_acc, label='Training', linewidth=2)
ax1.plot(val_acc, label='Validation', linewidth=2)
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
plt.savefig(report_dir+"/"+os.path.basename(os.path.normpath(report_dir))+".pdf", bbox_inches='tight')
plt.close()