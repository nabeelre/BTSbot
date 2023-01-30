import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
import json, datetime, os, sys

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import CNN_models
import bts_val
from create_pd_gr import create_train_data, only_pd_gr

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
})
plt.rcParams['axes.linewidth'] = 1.5

# /-----------------------------
#  HYPERPARAMETERS
# /-----------------------------

with open("train_config.json", 'r') as f:
    hparams = json.load(f)

loss = hparams['loss']
if hparams['optimizer']['kind'] == "Adam":
    opt_hparams = hparams['optimizer']
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=opt_hparams['learning_rate'], 
        beta_1=opt_hparams['beta_1'],
        beta_2=opt_hparams['beta_2'], 
        epsilon=None if opt_hparams['epsilon']=="None" else opt_hparams['epsilon'], 
        decay=opt_hparams['decay'],
        amsgrad=bool(opt_hparams['amsgrad'])
    )
weight_classes = bool(hparams['weight_classes'])
batch_size = hparams['batch_size']

tf.keras.backend.clear_session()

if bool(hparams['dont_use_GPU']):
    # DISABLE ALL GPUs
    tf.config.set_visible_devices([], 'GPU')

# removing negative diffs makes finding AGN easier
metadata_cols = hparams['metadata_cols']
metadata_cols.append('label')

# /-----------------------------
#  BASIC COMMAND LINE INTERFACE
# /-----------------------------

metadata = True if "meta" in sys.argv[1].lower() else False

N_max = hparams["N_max"]
epochs = hparams["epochs"]

if len(sys.argv) > 1:
    try: 
        model_type = getattr(CNN_models, sys.argv[1].lower())
    except:
        print("Could not find model of name", sys.argv[1].lower())
        exit(0)
    
if len(sys.argv) > 2:
    try:
        N_max = int(sys.argv[2])
        print("N_max overridden in command line as N_max =", sys.argv[2])
    except:
        print("Could not understand provided N_max override:", sys.argv[2])

if len(sys.argv) > 3:
    try:
        epochs = int(sys.argv[3])
        print("epochs overridden in command line as epochs =", sys.argv[3])
    except:
        print("Could not understand provided epocs override:", sys.argv[3])

patience = max(int(epochs*0.25), 50)

# /-----------------------------
#  LOAD TRAINING DATA
# /-----------------------------

if not (os.path.exists(f'data/candidates_v4_n{N_max}.csv') and 
        os.path.exists(f'data/triplets_v4_n{N_max}.npy')):
    create_train_data(['trues', 'dims', 'vars', 'MS'], only_pd_gr, N_max=N_max)
else:
    print("Training data already present")

df = pd.read_csv(f'data/candidates_v4_n{N_max}.csv')

if df.isnull().values.any():
    print("HAD TO REMOVE NANS")
    df = df.fillna(-999.0)

print(f'num_notbts: {np.sum(df.label == 0)}')
print(f'num_bts: {np.sum(df.label == 1)}')

triplets = np.load(f'data/triplets_v4_n{N_max}.npy', mmap_mode='r')
assert not np.any(np.isnan(triplets))

# /-----------------------------
#  TRAIN/VAL/TEST SPLIT
# /-----------------------------

test_split = 0.1  # fraction of all data
random_state = 2

ztfids_seen, ztfids_test = train_test_split(pd.unique(df['objectId']), test_size=test_split, random_state=random_state)

# Want array of indices for seen alerts and unseen/testing alerts
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

print(tf.config.list_physical_devices())

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

data_aug = hparams['data_aug']

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=bool(data_aug["h_flip"]),
    vertical_flip  =bool(data_aug["v_flip"]),
    fill_mode      =data_aug["fill_mode"],
    cval           =data_aug["cval"],
    preprocessing_function = rotate_incs_90 if bool(data_aug["rot"])  else None
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

if metadata:
    t_generator = train_datagen.flow(x_train, train_df, batch_size=batch_size, seed=random_state, shuffle=False)
    v_generator = val_datagen.flow(x_val, val_df, batch_size=batch_size, seed=random_state, shuffle=False)

    def multiinput_train_generator():
        while True:
            # get the data from the generator
            # data is [[img], [metadata and labels]]
            # yields batch_size number of entries
            data = t_generator.next()

            imgs = data[0]
            cols = data[1][:,:-1]
            targets = data[1][:,-1:]

            yield [imgs, cols], targets

    def multiinput_val_generator():
        while True:
            data = v_generator.next()

            imgs = data[0]
            cols = data[1][:,:-1]
            targets = data[1][:,-1:]

            yield [imgs, cols], targets

    training_generator = multiinput_train_generator()
    validation_generator = multiinput_val_generator()
else:
    training_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, seed=random_state, shuffle=False)
    validation_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size, seed=random_state, shuffle=False)

# /-----------------------------
#  OTHER MODEL SET UP
# /-----------------------------

if weight_classes:
    # weight data on number of examples per class?
    num_training_examples_per_class = np.array([len(y_train) - np.sum(y_train), np.sum(y_train)])
    assert 0 not in num_training_examples_per_class, 'found class without any examples!'

    # fewer examples -> larger weight
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
model_name = f"{model.name}-v4-n{N_max}{'-CPU' if bool(hparams['dont_use_GPU']) else ''}"

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
     'Training candids': list(df.candid[mask_train]),
     'Validation candids': list(df.candid[mask_val]),
     'Val misclassifications': misclassifications_val,
     'Training misclassifications': misclassifications_train,
     'Training history': h.history
     }
for k in report['Training history'].keys():
    report['Training history'][k] = np.array(report['Training history'][k]).tolist()

if metadata:
    report['metadata_cols'] = metadata_cols[:-1]

report_dir = "models/photoz/"+model_name+"/"+str(run_t_stamp)+"/"
model_dir = report_dir+"model/"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

f_name = os.path.join(report_dir, f'report.json')
with open(f_name, 'w') as f:
    json.dump(report, f, indent=2)

model.save(model_dir)
tf.keras.utils.plot_model(model, report_dir+"model_architecture.pdf", show_shapes=True, show_layer_names=False, show_layer_activations=True)

bts_val.run_val(report_dir, "train_config.json")