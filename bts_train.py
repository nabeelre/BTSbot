import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import wandb
import json
import sys
import os

from train_val_test_split import create_subset
import architectures
import bts_val

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
})
plt.rcParams['axes.linewidth'] = 1.5
random_state = 2

# /-----------------------------
#  HYPERPARAMETERS
# /-----------------------------


def sweep_train(config=None):
    with wandb.init(config=config) as run:
        train(run.config, run_name=run.name, sweeping=True)


def classic_train(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    train(config)


def train(config, run_name: str = None, sweeping: bool = False):
    if sys.platform == "darwin":
        # Disable GPUs if running on macOS
        print("disabling GPUs")
        tf.config.set_visible_devices([], 'GPU')

    loss = config['loss']
    if sys.platform == "darwin":
        print("Using keras legacy Adam optimizer for M1-chip compatibility")
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=config['learning_rate'],
            beta_1=config['beta_1'],
            beta_2=config['beta_2']
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config['learning_rate'],
            beta_1=config['beta_1'],
            beta_2=config['beta_2']
        )
    weight_classes = config['weight_classes']
    epochs = config["epochs"]
    patience = config['patience']
    random_state = config['random_seed']
    batch_size = config['batch_size']
    data_base_dir = config.get('data_base_dir', '')

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(random_state)

    try:
        N_max_p = config["N_maxs"][0]
        N_max_n = config["N_maxs"][1]
    except KeyError:
        N_max_p = config["N_max_p"]
        if "N_max_n" in config:
            N_max_n = config["N_max_n"]
        else:
            N_max_n = N_max_p

    if N_max_p == N_max_n:
        N_str = f"_N{N_max_p}"
    else:
        N_str = f"_Np{N_max_p}"
        if N_max_n:
            N_str += f"n{N_max_n}"

    try:
        crop_cutout_size = config['crop_cutout_size']
        if crop_cutout_size == 63:
            crop_cutout_str = ""
        else:
            crop_cutout_str = f"_c{crop_cutout_size}"
    except KeyError:
        crop_cutout_str = ""

    try:
        model_type = getattr(architectures, config['model_name'].lower())
    except AttributeError:
        print("Could not find model of name", config['model_name'].lower())
        exit(0)

    print(f"*** Running {model_type.__name__} with N_max_p={N_max_p}," +
          f"N_max_n={N_max_n}, and batch_size={batch_size} for epochs={epochs} ***")

    # /-----------------------------
    #  LOAD TRAINING DATA
    # /-----------------------------

    train_data_version = config['train_data_version']

    # enter new architectures and corresponding types here
    need_triplets = model_type.__name__ in [
        'mm_cnn', 'um_cnn', 'um_cnn_small', 'mm_cnn_small', 'mm_cnn_se'
    ]
    need_metadata = model_type.__name__ in [
        'mm_cnn', 'um_nn', 'mm_cnn_small', 'mm_cnn_se'
    ]

    triplets_path = (
        f'{data_base_dir}data/train_triplets_{train_data_version}'
        f'{N_str}{crop_cutout_str}.npy'
    )
    metadata_path = (
        f'{data_base_dir}data/train_cand_{train_data_version}{N_str}.csv'
    )
    triplets_present = os.path.exists(triplets_path)
    metadata_present = os.path.exists(metadata_path)

    if (need_triplets and (not triplets_present)) or (not metadata_present):
        print(f"Couldn't find {train_data_version}{N_str} train subset, creating...")
        # to fix: create_subset requires images even if only using um_nn
        create_subset("train", version_name=train_data_version,
                      N_max_p=N_max_p, N_max_n=N_max_n)
    else:
        print(f"{train_data_version} training data already present")

    cand = pd.read_csv(metadata_path)
    if need_triplets:
        triplets = np.load(triplets_path, mmap_mode='r')

    print(f'num_notbts: {np.sum(cand.label == 0)}')
    print(f'num_bts: {np.sum(cand.label == 1)}')

    if need_metadata:
        metadata_cols = config['metadata_cols']
        if cand[metadata_cols].isnull().values.any():
            print("Null in cand")
            exit(0)
    if need_triplets and np.any(np.isnan(triplets)):
        nan_trip_idxs = np.isnan(triplets).any(axis=(1, 2, 3))
        triplets = triplets[~nan_trip_idxs]

        cand = cand.drop(cand.index[nan_trip_idxs])
        cand.reset_index(inplace=True, drop=True)

        print("**** Null in triplets ****")
        print(f"Removed {np.sum(nan_trip_idxs)} alert(s)")

    # /-----------------------------
    #  LOAD VALIDATION DATA
    # /-----------------------------
    val_triplets_path = (
        f'{data_base_dir}data/val_triplets_{train_data_version}'
        f'{N_str}{crop_cutout_str}.npy'
    )
    val_metadata_path = (
        f'{data_base_dir}data/val_cand_{train_data_version}{N_str}.csv'
    )
    val_triplets_present = os.path.exists(val_triplets_path)
    val_metadata_present = os.path.exists(val_metadata_path)

    if (need_triplets and (not val_triplets_present)) or (not val_metadata_present):
        print(f"Couldn't find {train_data_version}{N_str} val subset, creating...")
        create_subset("val", version_name=train_data_version,
                      N_max_p=N_max_p, N_max_n=N_max_n)
    else:
        print(f"{train_data_version} val data already present")

    val_cand = pd.read_csv(val_metadata_path)
    if need_triplets:
        val_triplets = np.load(val_triplets_path, mmap_mode='r')

    # /----------------------------------
    #  MODEL INPUT AND SOME PARAMS PREP
    # /----------------------------------

    if need_triplets:
        x_train = triplets
        x_val = val_triplets
    else:
        x_train = cand[metadata_cols]
        x_val = val_cand[metadata_cols]

    y_train = cand['label']
    y_val = val_cand['label']

    print(f"{len(pd.unique(cand['objectId']))} train objects")
    print(f"{len(x_train)} train alerts")

    if "alert" in weight_classes:
        # weight data on number of ALERTS per class
        num_training_examples_per_class = np.array([
            np.sum(cand['label'] == 0), np.sum(cand['label'] == 1)
        ])
    elif "source" in weight_classes:
        # weight data on number of SOURCES per class
        num_training_examples_per_class = np.array([
            len(pd.unique(cand.loc[cand['label'] == 0, 'objectId'])),
            len(pd.unique(cand.loc[cand['label'] == 1, 'objectId']))
        ])
    else:
        # even weighting / no weighting
        num_training_examples_per_class = np.array([1, 1])

    # fewer examples -> larger weight
    weights_sum = np.sum(1 / num_training_examples_per_class)
    weights = (1 / num_training_examples_per_class) / weights_sum
    normalized_weight = weights / np.max(weights)

    class_weight = {i: w for i, w in enumerate(normalized_weight)}

    # image shape:
    if need_triplets:
        image_shape = x_train.shape[1:]
        print('Input image shape:', image_shape)

    # metadata shape (if necessary):
    if need_metadata:
        metadata_shape = (len(metadata_cols),)
        print('Input metadata shape:', metadata_shape)

    if need_triplets and need_metadata:
        model = model_type(config, image_shape=image_shape, metadata_shape=metadata_shape)
    elif need_triplets:
        model = model_type(config, image_shape=image_shape)
    elif need_metadata:
        model = model_type(config,  metadata_shape=metadata_shape)

    run_t_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = (
        f"{model.name}_{train_data_version}{N_str}"
        f"{''.join(['_CPU' if sys.platform == 'darwin' else ''])}"
    )

    # /-----------------------------
    #  SET UP CALLBACKS
    # /-----------------------------

    print(tf.config.list_physical_devices())

    # halt training if no improvement in validation loss over patience epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=1,
        patience=patience
    )

    # reduce learning rate if no improvement in validation loss over patience epochs
    LR_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=20,
        factor=config['reduce_LR_factor'],
        min_lr=config['reduce_LR_minLR'],
        verbose=0
    )

    if not sweeping:
        wandb.init(project="BTSbot")
        # Send parameters of this run to WandB
        for param in list(config):
            wandb.config[param] = config[param]

        run_name = wandb.run.name
    WandBLogger = wandb.keras.WandbMetricsLogger(log_freq=5)

    report_dir = f"models/{model_name}/{run_name}/"
    model_dir = report_dir+"best_model/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save new model whenever there's an improvement in val_loss
    checkpointing = tf.keras.callbacks.ModelCheckpoint(model_dir, verbose=1,
                                                       monitor="val_loss",
                                                       save_best_only=True)

    # /-----------------------------
    #  SET UP DATA GENERATORS WITH AUGMENTATION
    # /-----------------------------

    def rotate_incs_90(img):
        return np.rot90(img, np.random.choice([-1, 0, 1, 2]))

    if need_triplets:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=bool(config["data_aug_h_flip"]),
            vertical_flip=bool(config["data_aug_v_flip"]),
            fill_mode='constant',
            cval=0,
            preprocessing_function=rotate_incs_90 if bool(config["data_aug_rot"]) else None
        )

        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

        if need_metadata:
            # train_df is a combination of the desired metadata cols and y_train (labels)
            # we provide the model a custom generator function to separate these as necessary
            gen_cols = np.append(metadata_cols, ['label'])
            train_df = cand[gen_cols]
            val_df = val_cand[gen_cols]

            t_generator = train_datagen.flow(
                x_train, train_df, batch_size=batch_size,
                seed=random_state, shuffle=False
            )
            v_generator = val_datagen.flow(
                x_val, val_df, batch_size=batch_size,
                seed=random_state, shuffle=False
            )

            def multiinput_train_generator():
                while True:
                    # get the data from the generator
                    # data is [[img], [metadata and labels]]
                    # yields batch_size number of entries
                    data = t_generator.next()

                    imgs = data[0]
                    cols = data[1][:, :-1]
                    targets = data[1][:, -1:]

                    yield [imgs, cols], targets

            def multiinput_val_generator():
                while True:
                    data = v_generator.next()

                    imgs = data[0]
                    cols = data[1][:, :-1]
                    targets = data[1][:, -1:]

                    yield [imgs, cols], targets

            train_data = multiinput_train_generator()
            val_data = multiinput_val_generator()
        else:
            train_data = train_datagen.flow(
                x_train, y_train, batch_size=batch_size,
                seed=random_state, shuffle=False
            )
            val_data = val_datagen.flow(
                x_val, y_val, batch_size=batch_size,
                seed=random_state, shuffle=False
            )
    else:  # um_nn
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.shuffle(len(x_train)).batch(batch_size).repeat()

        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_data = val_data.shuffle(len(x_val)).batch(batch_size).repeat()

    # /-----------------------------
    #  COMPILE AND TRAIN MODEL
    # /-----------------------------

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    h = model.fit(
        train_data,
        steps_per_epoch=0.8*len(x_train) // batch_size,
        validation_data=val_data,
        validation_steps=(0.8*len(x_val)) // batch_size,
        class_weight=class_weight,
        epochs=epochs,
        verbose=2, callbacks=[early_stopping, LR_plateau, WandBLogger,
                              checkpointing]
    )

    # /-----------------------------
    #  LOG MISCLASSIFICATIONS
    # /-----------------------------

    print('Evaluating on training set to check misclassified samples:')
    if need_triplets and need_metadata:
        labels_training_pred = model.predict([x_train, train_df.iloc[:, :-1]],
                                             batch_size=batch_size, verbose=2)
    else:  # um_cnn or um_nn
        labels_training_pred = model.predict(x_train, batch_size=batch_size, verbose=2)

    # XOR will show misclassified samples
    misclassified_train_mask = np.array(list(map(int, cand.label))).flatten() ^ \
        np.array(list(map(int, np.rint(labels_training_pred)))).flatten()

    misclassified_train_mask = [ii for ii, mi in enumerate(misclassified_train_mask) if mi == 1]

    misclassifications_train = {int(c): [int(l), float(p)]
                                for c, l, p in zip(cand.candid.values[misclassified_train_mask],
                                                   cand.label.values[misclassified_train_mask],
                                                   labels_training_pred[misclassified_train_mask])}

    # /-----------------------------
    #  SAVE REPORT AND MODEL TO DISK
    # /-----------------------------

    # generate training report in json format
    print('Generating report...')
    report = {
        'Run time stamp': run_t_stamp,
        'Model name': model_name,
        'Run name': run_name,
        'Model trained': model_type.__name__,
        'Training data version': train_data_version,
        'Weighting loss contribution by class size': class_weight,
        'Train_config': dict(config),
        'Early stopping after epochs': patience,
        'Random state': random_state,
        'Number of training examples': x_train.shape[0],
        'Number of val examples': x_val.shape[0],
        'X_train shape': x_train.shape,
        'Y_train shape': y_train.shape,
        'X_val shape': x_val.shape,
        'Y_val shape': y_val.shape,
        'Misclassified training candids': list(misclassifications_train.keys()),
        'Training candids': list(cand.candid),
        'Validation candids': list(cand.candid),
        'Training misclassifications': misclassifications_train,
        'Training history': h.history
    }
    for k in report['Training history'].keys():
        report['Training history'][k] = np.array(report['Training history'][k]).tolist()

    f_name = os.path.join(report_dir, 'report.json')
    with open(f_name, 'w') as f:
        json.dump(report, f, indent=2)

    final_model_dir = report_dir+"final_model/"
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)

    model.save(final_model_dir)
    try:
        tf.keras.utils.plot_model(model, report_dir+"model_architecture.pdf",
                                  show_shapes=True, show_layer_names=False,
                                  show_layer_activations=True)
    except Exception as e:
        print(e)

    val_summary = bts_val.run_val(report_dir)

    wandb.summary['ROC_AUC'] = val_summary['roc_auc']
    wandb.summary['bal_acc'] = val_summary['bal_acc']
    wandb.summary['bts_acc'] = val_summary['bts_acc']
    wandb.summary['notbts_acc'] = val_summary['notbts_acc']

    wandb.summary['alert_precision'] = val_summary['alert_precision']
    wandb.summary['alert_recall'] = val_summary['alert_recall']
    alert_precision = val_summary['alert_precision']
    alert_recall = val_summary['alert_recall']
    if (alert_precision + alert_recall) > 0:
        wandb.summary['alert_F1'] = (
            2 * alert_precision * alert_recall
        ) / (alert_precision + alert_recall)
    else:
        wandb.summary['alert_F1'] = 0

    for policy_name in list(val_summary['policy_performance']):
        perf = val_summary['policy_performance'][policy_name]

        wandb.summary[policy_name+"_precision"] = perf['policy_precision']
        wandb.summary[policy_name+"_recall"] = perf['policy_recall']
        wandb.summary[policy_name+"_binned_precision"] = perf['binned_precision']
        wandb.summary[policy_name+"_binned_recall"] = perf['binned_recall']
        wandb.summary[policy_name+"_peakmag_bins"] = perf['peakmag_bins']

        wandb.summary[policy_name+"_save_dt"] = perf['med_save_dt']
        wandb.summary[policy_name+"_trigger_dt"] = perf['med_trigger_dt']

        policy_precision = perf['policy_precision']
        policy_recall = perf['policy_recall']
        if (policy_precision + policy_recall) > 0:
            wandb.summary[policy_name+"_F1"] = (
                2 * policy_precision * policy_recall
            ) / (policy_precision + policy_recall)
        else:
            wandb.summary[policy_name+"_F1"] = 0

    wandb.log({"figure": val_summary['fig']})


if __name__ == "__main__":
    if sys.argv[1] == "sweep":
        sweep_id = "gtpkgpkv"
        wandb.agent(sweep_id, function=sweep_train, count=15, project="BTSbot")
    else:
        classic_train(sys.argv[1])
