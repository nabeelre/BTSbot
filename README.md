<p align="center">
  <img
    src="https://github.com/nabeelre/BTSbot/assets/33795390/1b8586b1-5a89-4b84-a971-bf85fe722696"
    alt="BTSbot Logo"
    width="250px"
  />
</p>

[![DOI](https://zenodo.org/badge/517923027.svg)](https://zenodo.org/doi/10.5281/zenodo.10839684)
[![arXiv](https://img.shields.io/badge/Publication-2401.15167-b31b1b.svg)](https://arxiv.org/abs/2401.15167)
[![arXiv](https://img.shields.io/badge/ICML-2307.07618-b31b1b.svg)](https://arxiv.org/abs/2307.07618)

`BTSbot` is a multi-modal deep learning model to automate supernova identification and follow-up for the [Zwicky Transient Facility (ZTF)](https://www.ztf.caltech.edu) [Bright Transient Survey (BTS)](https://sites.astro.caltech.edu/ztf/bts/bts.php). 

`BTSbot` contributed to the first SN to be fully automatically discovered, confirmed, classified, and shared. ([AstroNote](https://www.wis-tns.org/astronotes/astronote/2023-265), [press release](https://news.northwestern.edu/stories/2023/10/first-supernova-detected-confirmed-classified-and-shared-by-ai/))

Presented at the [ML for Astrophysics workshop](https://ml4astro.github.io/icml2023/) at [ICML 2023](https://icml.cc/Conferences/2023) ([Extended abstract](https://arxiv.org/abs/2307.07618))

The training set for the production model is available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10839690).

Also see this [animated walkthrough](https://www.youtube.com/watch?v=qUwlQflDdEo) of the fully-automated BTS workflow

## A multi-modal convolutional neural network

![model](https://github.com/nabeelre/BTSbot/assets/33795390/c33431eb-2a0d-4ed1-8b30-11a5810699c4)

Fig. 5 from [Rehemtulla et al. 2024](https://arxiv.org/abs/2401.15167)

## Usage

Start with the usual imports.

```python
import tensorflow as tf
import numpy as np 
import pandas as pd
```

I've experienced weird behavior when training and running inference on the GPU cores of my M1 Mac, so we'll disable them here.

```python
import sys
if sys.platform == "darwin":
    tf.config.set_visible_devices([], 'GPU')
```

Load some example data. It contains alerts from two sources: ZTF23abhvlji (SN 2023tyk, a bright SNIa) and ZTF23abdsfms (AT 2023sxt, an average CV).

```python
cand = pd.read_csv("example_data/usage_candidates.csv", index_col=None)
trips = np.load("example_data/usage_triplets.npy", mmap_mode='r')
```

These are the metadata columns that the multi-modal `BTSbot` uses - order matters!

```python
metadata_cols = [
    "sgscore1", "distpsnr1", "sgscore2", "distpsnr2", "fwhm", "magpsf",
    "sigmapsf", "chipsf", "ra", "dec", "diffmaglim", "ndethist", "nmtchps",
    "age", "days_since_peak", "days_to_peak", "peakmag_so_far", "new_drb",
    "ncovhist", "nnotdet", "chinr", "sharpnr", "scorr", "sky", "maxmag_so_far"
]
```

First, unzip `BTSbot` at `production_models/v1.0.1.tar.gz` and then proceed with loading it.

```python
BTSbot = tf.keras.models.load_model("production_models/best_model/")
```

Now run `BTSbot` on the example alerts!

```python
raw_preds = BTSbot.predict([trips, cand[metadata_cols]], verbose=1)
```

Rearrange the scores and compare with the scores I get. You should get a number very close to zero - some minor deviation of scores is normal.

```python
raw_preds = np.transpose(raw_preds)[0]
print(np.median(np.abs(cand['expected_scores'] - raw_preds)))
```

Now `BTSbot` is up and running! If you have access to `Kowalski` you can query for new sources to run `BTSbot` on using `download_training_data()`; if not, see `alert_utils()` for functions to process raw triplets and compute metadata features as `BTSbot` expects them.

## Performance

`BTSbot` finds nearly all SNe of interest from the input data stream (~100% completeness) with little contamination from uninteresting phenomena (93% purity) and does so as quickly as humans typically do.

![test_performance.pdf](https://github.com/nabeelre/BTSbot/files/15135081/test_performance.pdf)

Fig. 7 from [Rehemtulla et al. 2024](https://arxiv.org/abs/2401.15167)

## Citing `BTSbot`

If you use or reference `BTSbot` please cite [Rehemtulla et al. 2024](https://arxiv.org/abs/2401.15167) ([ADS](https://ui.adsabs.harvard.edu/abs/2024arXiv240115167R/abstract))
