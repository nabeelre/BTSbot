import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
import json, datetime, os, sys

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import CNN_models
import bts_sweep_val
from manage_data import create_subset, only_pd_gr

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


def train_sweep_iter(config=None):
    with wandb.init(config=config):
        config = wandb.config
        plt.rcParams.update({
            "font.family": "Times New Roman",
            "font.size": 14,
        })
        plt.rcParams['axes.linewidth'] = 1.5
        random_state = 2

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(random_state)

        # /-----------------------------
        #  HYPERPARAMETERS
        # /-----------------------------

        loss = config['loss']
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config['learning_rate'], 
            beta_1=config['beta_1'],
            beta_2=config['beta_2'], 
        )
        weight_classes = True
        epochs = config["epochs"]
        patience = config["patience"]


        if sys.platform == "darwin":
            # DISABLE ALL GPUs
            print("Disabling GPU")
            tf.config.set_visible_devices([], 'GPU')

        # removing negative diffs makes finding AGN easier
        metadata_cols = config['metadata_cols']
        all_cols = np.append(metadata_cols, ['label'])

        metadata = True if "meta" in config['model_name'] else False

        try: 
            model_type = getattr(CNN_models, config['model_name'])
        except:
            print("Could not find model of name", config['model_name'])
            exit(0)

        # /-----------------------------
        #  LOAD TRAINING DATA
        # /-----------------------------

        if not (os.path.exists(f'data/train_cand_v5_n{config["N_max"]}.csv') and 
                os.path.exists(f'data/train_triplets_v5_n{config["N_max"]}.npy')):
            print("Couldn't find correct train data subset, creating...")
            create_subset("train", N_max=config["N_max"])
        else:
            print("Training data already present")

        cand = pd.read_csv(f'data/train_cand_v5_n{config["N_max"]}.csv')

        if cand.isnull().values.any():
            print("HAD TO REMOVE NANS")
            cand = cand.fillna(-999.0)

        print(f'num_notbts: {np.sum(cand.label == 0)}')
        print(f'num_bts: {np.sum(cand.label == 1)}')

        triplets = np.load(f'data/train_triplets_v5_n{config["N_max"]}.npy', mmap_mode='r')
        assert not np.any(np.isnan(triplets))

        # /-----------------------------
        #  LOAD VALIDATION DATA
        # /-----------------------------

        val_cand = pd.read_csv(f'data/val_cand_v5.csv')
        val_triplets = np.load(f'data/val_triplets_v5.npy', mmap_mode='r')

        # /-----------------------------
        #  PREP DATA AS MODEL INPUT
        # /-----------------------------

        x_train, y_train = triplets, cand['label']
        x_val, y_val = val_triplets, val_cand['label']

        # train_df is a combination of the desired metadata and y_train (labels)
        # we provide the model a custom generator function to separate these as necessary
        train_df = cand[all_cols]
        val_df = val_cand[all_cols]

        print(f"{len(pd.unique(cand['objectId']))} train objects")
        print(f"{len(x_train)} train alerts")

        # /-----------------------------
        #  SET UP CALLBACKS
        # /-----------------------------

        print(tf.config.list_physical_devices())

        # halt training if no gain in validation accuracy over patience epochs
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1
        )

        LR_plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            patience=20,
            factor=config['reduce_LR_factor'],
            min_lr=config['reduce_LR_minLR'],
            verbose=1
        )

        WandBLogger = WandbMetricsLogger(log_freq=5)

        # WandBCheckpoints = WandbModelCheckpoint("models")

        # tensorboard = tf.keras.callbacks.TensorBoard(
        #     log_dir="tb_logs/", 
        #     histogram_freq=1,
        #     write_graph=True,
        #     write_images=True,
        #     update_freq='epoch',
        #     profile_batch=1
        # )

        # /-----------------------------
        #  SET UP DATA GENERATORS WITH AUGMENTATION
        # /-----------------------------

        def rotate_incs_90(img):
            return np.rot90(img, np.random.choice([-1, 0, 1, 2]))

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=bool(config["data_aug_h_flip"]),
            vertical_flip  =bool(config["data_aug_v_flip"]),
            fill_mode      ='constant',
            cval           =0,
            preprocessing_function = rotate_incs_90 if bool(config["data_aug_rot"])  else None
        )

        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

        if metadata:
            t_generator = train_datagen.flow(x_train, train_df, batch_size=config['batch_size'], seed=random_state, shuffle=False)
            v_generator = val_datagen.flow(x_val, val_df, batch_size=config['batch_size'], seed=random_state, shuffle=False)

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
            training_generator = train_datagen.flow(x_train, y_train, batch_size=config['batch_size'], seed=random_state, shuffle=False)
            validation_generator = val_datagen.flow(x_val, y_val, batch_size=config['batch_size'], seed=random_state, shuffle=False)

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
            model = model_type(config, image_shape=image_shape, metadata_shape=metadata_shape)
        else:
            model = model_type(config, image_shape=image_shape)

        run_t_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model.name}-v5-n{config['N_max']}-bs{config['batch_size']}{'-CPU' if sys.platform == 'darwin' else ''}"

        # /-----------------------------
        #  COMPILE AND TRAIN MODEL
        # /-----------------------------

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        h = model.fit(
            training_generator,
            steps_per_epoch=0.8*len(x_train) // config['batch_size'],
            validation_data=validation_generator,
            validation_steps=(0.8*len(x_val)) // config['batch_size'],
            class_weight=class_weight,
            epochs=epochs,
            verbose=1, callbacks=[early_stopping, LR_plateau, WandBLogger]
        )

        # /-----------------------------
        #  LOG MISCLASSIFICATIONS
        # /-----------------------------

        print('Evaluating on training set to check misclassified samples:')
        if metadata:
            labels_training_pred = model.predict([x_train, train_df.iloc[:,:-1]], batch_size=config['batch_size'], verbose=1)
        else:
            labels_training_pred = model.predict(x_train, batch_size=config['batch_size'], verbose=1)

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
        report = {'Run time stamp': run_t_stamp,
            'Model name': model_name,
            'Model trained': model_type.__name__,
            'Train_config': {
                "N_max": config['N_max'],
                "epochs": config['epochs'],
                "loss": config['loss'],
                "optimizer": {
                    "kind": "Adam",
                    "learning_rate": config['learning_rate'],
                    "beta_1": config['beta_1'],
                    "beta_2": config['beta_2'],
                    "epsilon": "None",
                    "decay": 0.0,
                    "amsgrad": 0
                },
                "weight_classes": 1,
                "batch_size": config['batch_size'],
                "metadata_cols": config['metadata_cols'],
                "data_aug": {
                    "h_flip": 1,
                    "v_flip": 1,
                    "fill_mode": "constant",
                    "cval": 0,
                    "rot": 1
                },
                "val_cuts": {
                    "sne_only": config['val_sne_only'],
                    "keep_near_threshold": config['val_keep_near_threshold'],
                    "rise_only": config['val_rise_only']
                }
            },
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

        if metadata:
            report['metadata_cols'] = metadata_cols[:-1]

        report_dir = "models/"+model_name+"/"+str(run_t_stamp)+"/"
        model_dir = report_dir+"model/"

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        f_name = os.path.join(report_dir, f'report.json')
        with open(f_name, 'w') as f:
            json.dump(report, f, indent=2)

        model.save(model_dir)
        tf.keras.utils.plot_model(model, report_dir+"model_architecture.pdf", show_shapes=True, show_layer_names=False, show_layer_activations=True)

        val_summary = bts_sweep_val.run_val(report_dir, config)

        wandb.summary['ROC_AUC']    = val_summary['roc_auc']
        wandb.summary['bal_acc']    = val_summary['bal_acc']
        wandb.summary['bts_acc']    = val_summary['bts_acc']
        wandb.summary['notbts_acc'] = val_summary['notbts_acc']
        wandb.summary['precision']  = val_summary['precision']
        wandb.summary['recall']     = val_summary['recall']

        for name, precision, recall in zip(val_summary['metric_names'], 
                                        val_summary['metric_precision'], 
                                        val_summary['metric_recall']):
            wandb.summary[name+"_precision"] = precision
            wandb.summary[name+"_recall"] = recall


if __name__ == "__main__":
    testing_sweep_id = "mttelhfy"
    sweep_id = "er9c8y7g"
    wandb.agent(testing_sweep_id, function=train_sweep_iter, count=int(sys.argv[1]), project="BNB-classifier")
