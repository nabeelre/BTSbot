import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json, datetime, os
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, MaxPooling2D, Flatten, Dropout
from astropy.stats import sigma_clipped_stats
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from pandas.plotting import register_matplotlib_converters, scatter_matrix

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
})
plt.rcParams['axes.linewidth'] = 1.5


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


def vgg6(input_shape=(63, 63, 3), n_classes: int = 1):
    """
        VGG6
    :param input_shape:
    :param n_classes:
    :return:
    """

    model = tf.keras.models.Sequential(name='VGG6')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, name='conv1'))
    model.add(Conv2D(16, (3, 3), activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25, name='drop_0.25'))

    model.add(Conv2D(32, (3, 3), activation='relu', name='conv3'))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv4'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25, name='drop2_0.25'))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', name='fc_1'))
    model.add(Dropout(0.4, name='drop3_0.4'))
    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(Dense(n_classes, activation=activation, name='fc_out'))

    return model


def vgg16(input_shape=(63, 63, 3), n_classes: int = 1):
    # https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    
    model = tf.keras.models.Sequential(name='VGG16')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), padding="same", activation="relu", name='conv1'))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", name='conv2'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", name='conv3'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", name='conv4'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name='conv5'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name='conv6'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name='conv7'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv8'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv9'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv10'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv11'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv12'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv13'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(Dense(units=n_classes, activation=activation))
    
    return model


def vgg_thin_reduce(input_shape=(63, 63, 3), n_classes: int = 1):    
    model = tf.keras.models.Sequential(name='VGG16')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(input_shape=input_shape, filters=16, kernel_size=(3,3), padding="same", activation="relu", name='conv1'))
    model.add(Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu", name='conv2'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", name='conv3'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", name='conv4'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", name='conv5'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", name='conv6'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", name='conv7'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv8'))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv9'))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv10'))
#     model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv11'))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv12'))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv13'))
#     model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(units=512,activation="relu"))
#     model.add(Dense(units=4096,activation="relu"))
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(Dense(units=n_classes, activation=activation))
    
    return model

N_max = 10
loss = 'binary_crossentropy'
optimizer = 'adam'
epochs = 100
patience = 500
class_weight = True
batch_size = 64

# create_train_data(['bts_true', 'bts_false', 'MS'], only_pd_gr, name="pd_gr", N_max=N_max)
df = pd.read_csv(f'data/candidates_pd_gr_{N_max}max.csv')
# display(df)
# df.info()
# df.describe()
print(f'num_notbts: {np.sum(df.label == 0)}')
print(f'num_bts: {np.sum(df.label == 1)}')

# We will use memory mapping as the file is relatively large (1 GB)
triplets = np.load(f'data/triplets_pd_gr_{N_max}max.npy', mmap_mode='r')

# /-----------------------
test_split = 0.1  # fraction of all data
random_state = 2

ztfids_seen, ztfids_test = train_test_split(pd.unique(df['objectId']), test_size=test_split, random_state=random_state)

# Want array of indices for training alerts and testing alerts
# Need to shuffle because validation is bottom 10% of train - shuffle test as well for consistency
is_seen = df['objectId'].isin(ztfids_seen)
is_test = ~is_seen
mask_seen = shuffle(df.index.values[is_seen], random_state=random_state)
mask_test  = shuffle(df.index.values[is_test], random_state=random_state)

x_seen, y_seen = triplets[mask_seen], df['label'][mask_seen]
x_test,  y_test  = triplets[mask_test] , df['label'][mask_test]

num_seen_obj = len(ztfids_seen)
num_test_obj = len(ztfids_test)
num_obj = len(pd.unique(df['objectId']))
print(f"{num_seen_obj} seen/train+val objects")
print(f"{num_test_obj} unseen/test objects")
print(f"{100*(num_seen_obj/num_obj):.2f}%/{100*(num_test_obj/num_obj):.2f}% seen/unseen split by object\n")

num_seen_alr = len(x_seen)
num_test_alr = len(x_test)
num_alr = len(df['objectId'])
print(f"{num_seen_alr} seen/train+val alerts")
print(f"{num_test_alr} unseen/test alerts")
print(f"{100*(num_seen_alr/num_alr):.2f}%/{100*(num_test_alr/num_alr):.2f}% seen/unseen split by alert\n")

# /-----------------------
validation_split = 0.1  # fraction of the seen data

ztfids_train, ztfids_val = train_test_split(ztfids_seen, test_size=validation_split, random_state=random_state)

is_train = df['objectId'].isin(ztfids_train)
is_val = df['objectId'].isin(ztfids_val)
mask_train = shuffle(df.index.values[is_train], random_state=random_state)
mask_val  = shuffle(df.index.values[is_val], random_state=random_state)

x_train, y_train = triplets[mask_train], df['label'][mask_train]
x_val, y_val = triplets[mask_val], df['label'][mask_val]

num_train_obj = len(ztfids_train)
num_val_obj = len(ztfids_val)
num_obj = len(pd.unique(df['objectId']))
print(f"{num_train_obj} train objects")
print(f"{num_val_obj} val objects")
print(f"{100*(num_train_obj/num_obj):.2f}%/{100*(num_val_obj/num_obj):.2f}% train/val split by object\n")

num_train_alr = len(x_train)
num_val_alr = len(x_val)
num_alr = len(df['objectId'])
print(f"{num_train_alr} train alerts")
print(f"{num_val_alr} val alerts")
print(f"{100*(num_train_alr/num_alr):.2f}%/{100*(num_val_alr/num_alr):.2f}% train/val split by alert\n")

# /-----------------------
def save_report(path: str = './', stamp: str = None, report: dict = dict()):
    f_name = os.path.join(path, f'report.{stamp}.json')
    with open(f_name, 'w') as f:
        json.dump(report, f, indent=2)

masks = {'training': mask_train, 'val': mask_val, 'test': mask_test}

# /-----------------------
tf.keras.backend.clear_session()

# halt training if no gain in validation accuracy over patience epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

data_augmentation = {'horizontal_flip': True,
                     'vertical_flip': True,
                     'rotation_range': 0,  # should rotate !!!
                     'fill_mode': 'constant',
                     'cval': 1e-9}
datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=data_augmentation['horizontal_flip'],
                                                          vertical_flip  =data_augmentation['vertical_flip'],
                                                          rotation_range =data_augmentation['rotation_range'],
                                                          fill_mode      =data_augmentation['fill_mode'],
                                                          cval           =data_augmentation['cval'])

training_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
validation_generator = datagen.flow(x_val, y_val, batch_size=batch_size)

# /-----------------------
binary_classification = True if loss == 'binary_crossentropy' else False
n_classes = 1 if binary_classification else 2

# training data weights
if class_weight:
    # weight data class depending on number of examples?
    if not binary_classification:
        num_training_examples_per_class = np.sum(y_train, axis=0)
    else:
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

# /-----------------------
model = vgg6(input_shape=image_shape, n_classes=n_classes)

# set up optimizer:
if optimizer == 'adam':
    optimzr = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.9, beta_2=0.999,
                                       epsilon=None, decay=0.0, amsgrad=False)
elif optimizer == 'sgd':
    optimzr = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=1e-6, nesterov=True)
else:
    print('Could not recognize optimizer, using Adam')
    optimzr = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.9, beta_2=0.999,
                                       epsilon=None, decay=0.0, amsgrad=False)

assert not np.any(np.isnan(triplets))
model.compile(optimizer=optimzr, loss=loss, metrics=['accuracy'])

# /-----------------------
run_t_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f'{model.name}_{run_t_stamp}'

h = model.fit(training_generator,
              steps_per_epoch=0.8*len(x_train) // batch_size,
              validation_data=validation_generator,
              validation_steps=(0.8*len(x_test)) // batch_size,
              class_weight=class_weight,
              epochs=epochs,
              verbose=1, callbacks=[early_stopping])

# /-----------------------
print('Evaluating on training set to check misclassified samples:')
labels_training_pred = model.predict(x_train, batch_size=batch_size, verbose=1)
# XOR will show misclassified samples
misclassified_train_mask = np.array(list(map(int, df.label[masks['training']]))).flatten() ^ \
                           np.array(list(map(int, np.rint(labels_training_pred)))).flatten()

misclassified_train_mask = [ii for ii, mi in enumerate(misclassified_train_mask) if mi == 1]

misclassifications_train = {int(c): [int(l), float(p)]
                            for c, l, p in zip(df.candid.values[masks['training']][misclassified_train_mask],
                                               df.label.values[masks['training']][misclassified_train_mask],
                                               labels_training_pred[misclassified_train_mask])}
# print(misclassifications_train)

print('Evaluating on validation set for loss and accuracy:')
preds = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=1)
val_loss = float(preds[0])
val_accuracy = float(preds[1])
print("Loss = " + str(val_loss))
print("Val Accuracy = " + str(val_accuracy))

print('Evaluating on validation set to check misclassified samples:')
preds = model.predict(x=x_val, batch_size=batch_size, verbose=1)

# XOR will show misclassified samples
misclassified_val_mask = np.array(list(map(int, df.label[masks['val']]))).flatten() ^ \
                          np.array(list(map(int, np.rint(preds)))).flatten()
misclassified_val_mask = [ii for ii, mi in enumerate(misclassified_val_mask) if mi == 1]

misclassifications_val = {int(c): [int(l), float(p)]
                           for c, l, p in zip(df.candid.values[masks['val']][misclassified_val_mask],
                                              df.label.values[masks['val']][misclassified_val_mask],
                                              preds[misclassified_val_mask])}

# round probs to nearest int (0 or 1)
labels_pred = np.rint(preds)

# /-----------------------
report_dir = "models/"+model_name+"/"
model_dir = report_dir+"model/"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

fpr, tpr, thresholds = roc_curve(df['label'][masks['val']], preds)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
CFD = ConfusionMatrixDisplay.from_predictions(df.label.values[masks['val']], 
                                        labels_pred, normalize='true', ax=ax)
plt.close()

# /-----------------------
val_labels = np.array(list(map(int, df.label[masks['val']]))).flatten()
val_rpreds = np.array(list(map(int, np.rint(preds)))).flatten()

val_TP_mask = np.bitwise_and(val_labels, val_rpreds)
val_TN_mask = 1-(np.bitwise_or(val_labels, val_rpreds))
val_FP_mask = np.bitwise_and(1-val_labels, val_rpreds)
val_FN_mask = np.bitwise_and(val_labels, 1-val_rpreds)

val_TP_idxs = [ii for ii, mi in enumerate(val_TP_mask) if mi == 1]
val_TN_idxs = [ii for ii, mi in enumerate(val_TN_mask) if mi == 1]
val_FP_idxs = [ii for ii, mi in enumerate(val_FP_mask) if mi == 1]
val_FN_idxs = [ii for ii, mi in enumerate(val_FN_mask) if mi == 1]

# per object model accuracy for val objects in g, r bands
val_perobj_g_acc = np.zeros(len(ztfids_val))
val_perobj_r_acc = np.zeros(len(ztfids_val))

for i, ztfid in enumerate(ztfids_val): 
    cands = df[df['objectId']==ztfid]
    label = cands['label'].to_numpy()[0]

    g_cands = cands[cands['fid']==1]
    g_trips = triplets[df['objectId']==ztfid][cands['fid']==1]

    r_cands = cands[cands['fid']==2]
    r_trips = triplets[df['objectId']==ztfid][cands['fid']==2]

    if len(g_cands) > 0:
        g_preds = np.array(np.rint(model.predict(g_trips).flatten()), dtype=int)
        val_perobj_g_acc[i] = np.sum(g_preds==label)/len(g_trips)
    else:
        g_preds = []
        val_perobj_g_acc[i] = -1
     
    if len(r_cands) > 0:
        r_preds = np.array(np.rint(model.predict(r_trips).flatten()), dtype=int)
        val_perobj_r_acc[i] = np.sum(r_preds==label)/len(r_trips)
    else:
        r_preds = []
        val_perobj_r_acc[i] = -1

# /-----------------------
# generate training report in json format
print('Generating report...')
r = {'Run time stamp': run_t_stamp,
     'Model name': model_name,
     'Model trained': 'vgg6',
     'Batch size': batch_size,
     'Optimizer': optimizer,
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
     'Data augmentation': data_augmentation,
     'Confusion matrix': CFD.confusion_matrix.tolist(),
     'Misclassified val candids': list(misclassifications_val.keys()),
     'Misclassified training candids': list(misclassifications_train.keys()),
     'Val misclassifications': misclassifications_val,
     'Training misclassifications': misclassifications_train,
     'Training history': h.history
     }
for k in r['Training history'].keys():
    r['Training history'][k] = np.array(r['Training history'][k]).tolist()

# print(r)

save_report(path=report_dir, stamp=run_t_stamp, report=r)
model.save(model_dir)

with open(report_dir+'model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'), expand_nested=True, show_trainable=True)

# /-----------------------

if 'accuracy' in h.history:
    train_acc = h.history['accuracy']
    val_acc = h.history['val_accuracy']
else:
    train_acc = h.history['acc']
    val_acc = h.history['val_acc']

train_loss = h.history['loss']
val_loss = h.history['val_loss']

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), dpi=250)

ax1.plot(train_acc, label='Training', linewidth=2)
ax1.plot(val_acc, label='Validation', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='best')
ax1.set_ylim([0.6,0.9])
ax1.grid(True, linewidth=.3)

ax2.plot(train_loss, label='Training', linewidth=2)
ax2.plot(val_loss, label='Validation', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='best')
ax2.set_ylim([0.2,0.7])
ax2.grid(True, linewidth=.3)

bins = np.arange(12,22,0.5)
ax3.hist(df['magpsf'][masks['val']].to_numpy()[val_TP_idxs], histtype='step', color='g', linewidth=2, label='TP', bins=bins, zorder=2)
ax3.hist(df['magpsf'][masks['val']].to_numpy()[val_TN_idxs], histtype='step', color='b', linewidth=2, label='TN', bins=bins, zorder=3)
ax3.hist(df['magpsf'][masks['val']].to_numpy()[val_FP_idxs], histtype='step', color='r', linewidth=2, label='FP', bins=bins, zorder=4)
ax3.hist(df['magpsf'][masks['val']].to_numpy()[val_FN_idxs], histtype='step', color='orange', linewidth=2, label='FN', bins=bins, zorder=5)
ax3.axvline(18.5, c='k', linewidth=2, linestyle='dashed', label='18.5', alpha=0.5, zorder=10)
ax3.legend(loc='upper left')
ax3.set_xlabel('Magnitude')
ax3.set_ylim([0,int(len(x_val)/5)])
ax3.xaxis.set_minor_locator(MultipleLocator(1))

ax4.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.plot(fpr, tpr, lw=2, label=f'ROC (area = {roc_auc:.5f})')
ax4.set_xlabel('False Positive Rate (Contamination)')
ax4.set_ylabel('True Positive Rate (Sensitivity)')
ax4.legend(loc="lower right")
ax4.grid(True, linewidth=.3)
ax4.set(aspect='equal')

ConfusionMatrixDisplay.from_predictions(df.label.values[masks['val']], 
                                        labels_pred, normalize='true', 
                                        display_labels=["notBTS", "BTS"], 
                                        cmap=plt.cm.Blues, colorbar=False, ax=ax5)

hist, xbins, ybins, im = ax6.hist2d(val_perobj_g_acc, val_perobj_r_acc, norm=LogNorm(), bins=4, range=[[0,1],[0,1]])
ax6.set_xlabel('Per-object g-band accuracy')
ax6.set_ylabel('Per-object r-band accuracy')
ax6.set(aspect='equal')

for i in range(len(ybins)-1):
    for j in range(len(xbins)-1):
        ax6.text(xbins[j]+0.125,ybins[i]+0.125, int(hist.T[i,j]), color="w", ha="center", va="center", fontweight="bold")

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.tick_params(which='both', width=1.5)

plt.tight_layout()
plt.savefig(report_dir+"fig.pdf", bbox_inches='tight')
plt.show()