import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
import json, datetime, os, sys
import wandb

import CNN_models
import bts_val
from manage_data import create_subset

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


def train(config, run_name : str = None, sweeping : bool = False):
    loss = config['loss']
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['learning_rate'], 
        beta_1=config['beta_1'],
        beta_2=config['beta_2']
    )
    weight_classes = config['weight_classes']
    epochs = config["epochs"]
    patience = config['patience']
    random_state = config['random_seed']

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(random_state)

    if sys.platform == "darwin":
        # Disable GPUs if running on macOS
        tf.config.set_visible_devices([], 'GPU')

    N_max = config["N_max"]
    batch_size = config['batch_size']

    try: 
        model_type = getattr(CNN_models, config['model_name'].lower())
    except:
        print("Could not find model of name", sys.argv[1].lower())
        exit(0)
    metadata = True if len(config['metadata_cols']) > 0 else False
        
    # /-----------------------------
    #  BASIC COMMAND LINE INTERFACE
    # /-----------------------------

    if not sweeping:
        if len(sys.argv) > 2:
            try:
                N_max = int(sys.argv[2])
                config['N_max'] = N_max
                print("N_max overridden in command line as N_max =", sys.argv[2])
            except:
                print("Could not understand provided N_max override:", sys.argv[2])

        if len(sys.argv) > 3:
            try:
                batch_size = int(sys.argv[3])
                config['batch_size'] = batch_size
                print("batch_size overridden in command line as batch_size =", sys.argv[3])
            except:
                print("Could not understand provided batch_size override:", sys.argv[3])

    print(f"*** Running {model_type.__name__} with N_max={N_max} and batch_size={batch_size} for epochs={epochs} ***")

    # /-----------------------------
    #  LOAD TRAINING DATA
    # /-----------------------------

    train_data_version = config['train_data_version']

    if not (os.path.exists(f'data/train_cand_{train_data_version}_n{N_max}.csv') and 
            os.path.exists(f'data/train_triplets_{train_data_version}_n{N_max}.npy')):
        print("Couldn't find correct train data subset, creating...")
        create_subset("train", N_max=N_max)
    else:
        print("Training data already present")

    cand = pd.read_csv(f'data/train_cand_{train_data_version}_n{N_max}.csv')
    triplets = np.load(f'data/train_triplets_{train_data_version}_n{N_max}.npy', mmap_mode='r')

    print(f'num_notbts: {np.sum(cand.label == 0)}')
    print(f'num_bts: {np.sum(cand.label == 1)}')

    if cand[config['metadata_cols']].isnull().values.any():
        print("Null in cand")
        exit(0)
    if np.any(np.isnan(triplets)):
        print("Null in triplets")
        exit(0)

    # /-----------------------------
    #  LOAD VALIDATION DATA
    # /-----------------------------

    val_cand = pd.read_csv(f'data/val_cand_{train_data_version}.csv')
    val_triplets = np.load(f'data/val_triplets_{train_data_version}.npy', mmap_mode='r')

    # /-----------------------------
    #  PREP DATA AS MODEL INPUT
    # /-----------------------------

    gen_cols = np.append(config['metadata_cols'], ['label'])

    x_train, y_train = triplets, cand['label']
    x_val, y_val = val_triplets, val_cand['label']

    # train_df is a combination of the desired metadata cols and y_train (labels)
    # we provide the model a custom generator function to separate these as necessary
    train_df = cand[gen_cols]
    val_df = val_cand[gen_cols]

    print(f"{len(pd.unique(cand['objectId']))} train objects")
    print(f"{len(x_train)} train alerts")

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

    if "alert" in weight_classes:
        # weight data on number of ALERTS per class
        num_training_examples_per_class = np.array([np.sum(cand['label'] == 0), np.sum(cand['label'] == 1)])
    elif "source" in weight_classes:
        # weight data on number of SOURCES per class
        num_training_examples_per_class = np.array([len(pd.unique(cand.loc[cand['label'] == 0, 'objectId'])),
                                                    len(pd.unique(cand.loc[cand['label'] == 1, 'objectId']))])
    else:
        # even weighting / no weighting
        num_training_examples_per_class = np.array([1,1])

    # fewer examples -> larger weight
    weights = (1 / num_training_examples_per_class) / np.linalg.norm((1 / num_training_examples_per_class))
    normalized_weight = weights / np.max(weights)

    class_weight = {i: w for i, w in enumerate(normalized_weight)}

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
    model_name = f"{model.name}_{train_data_version}_n{N_max}{'_CPU' if sys.platform == 'darwin' else ''}"

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
        verbose=1, callbacks=[early_stopping, LR_plateau, WandBLogger]
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

    report_dir = f"models/{model_name}/{run_name}/"
    model_dir = report_dir+"model/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    f_name = os.path.join(report_dir, f'report.json')
    with open(f_name, 'w') as f:
        json.dump(report, f, indent=2)

    model.save(model_dir)
    tf.keras.utils.plot_model(model, report_dir+"model_architecture.pdf", show_shapes=True, show_layer_names=False, show_layer_activations=True)

    val_summary = bts_val.run_val(report_dir)

    wandb.summary['ROC_AUC']    = val_summary['roc_auc']
    wandb.summary['bal_acc']    = val_summary['bal_acc']
    wandb.summary['bts_acc']    = val_summary['bts_acc']
    wandb.summary['notbts_acc'] = val_summary['notbts_acc']

    wandb.summary['alert_precision'] = val_summary['alert_precision']
    wandb.summary['alert_recall'] = val_summary['alert_recall']
    wandb.summary['alert_F1'] = (2 * val_summary['alert_precision'] * val_summary['alert_recall']) / (val_summary['alert_precision'] + val_summary['alert_recall'])

    for policy_name in list(val_summary['policy_performance']):
        perf = val_summary['policy_performance'][policy_name]

        wandb.summary[policy_name+"_overall_precision"] = perf['overall_precision']
        wandb.summary[policy_name+"_overall_recall"] = perf['overall_recall']
        wandb.summary[policy_name+"_precision"] = perf['precision']
        wandb.summary[policy_name+"_recall"] = perf['recall']

        wandb.summary[policy_name+"_med_del_st"] = perf['med_del_st']

        wandb.summary[policy_name+"_F1"] = (2 * perf['overall_precision'] * perf['overall_recall']) / (perf['overall_precision'] + perf['overall_recall'])

if __name__ == "__main__":
    if sys.argv[1] == "sweep":
        sweep_id = "m1il156a"
        wandb.agent(sweep_id, function=sweep_train, count=10, project="BTSbot")
    else:
        classic_train(sys.argv[1])
