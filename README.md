<p align="center">
  <img
    src="https://github.com/nabeelre/BTSbot/assets/33795390/1b8586b1-5a89-4b84-a971-bf85fe722696"
    alt="BTSbot Logo"
    width="250px"
  />
</p>

`BTSbot` is a machine learning tool to automate source identification and follow-up for the [Zwicky Transient Facility (ZTF)](https://www.ztf.caltech.edu) [Bright Transient Survey (BTS)](https://sites.astro.caltech.edu/ztf/bts/bts.php). ZTF is a robotic observatory that looks for changes in the night sky by repeatedly taking new images and comparing them with historical ones. ZTF finds thousands of supernovae (SNe; the explosive deaths of stars) each year, and BTS endeavors to classify them with follow-up observations and build a large, complete sample of SNe. `BTSbot` automatically looks through ZTF data to find the SNe of interest to BTS and send follow-up observation requests for them. [Fritz](https://github.com/fritz-marshal/fritz)/[SkyPortal](https://github.com/skyportal/skyportal) and [Kowalski](https://github.com/skyportal/kowalski) aid in coordinating between ZTF's data stream and `BTSbot`

Presented at the [ML for Astrophysics workshop](https://ml4astro.github.io/icml2023/) at [ICML 2023](https://icml.cc/Conferences/2023) ([Extended abstract](https://arxiv.org/abs/2307.07618))

`BTSbot` contributed to the first SN to be fully automatically discovered, confirmed, classified, and shared. ([AstroNote](https://www.wis-tns.org/astronotes/astronote/2023-265), [press release](https://news.northwestern.edu/stories/2023/10/first-supernova-detected-confirmed-classified-and-shared-by-ai/))

The training set for the production model is available [here](https://nuwildcat.sharepoint.com/:f:/s/WNB-MillerAstro/EnYj8QeyvM5BoMbBp2YK5TgBRIwz4F8nHLsUZkSwnGDk0A), but this only includes public alerts (i.e. those with programid=1).

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

![test_performance](https://github.com/nabeelre/BTSbot/assets/33795390/c9d1b930-c980-4e59-95d1-074bf7e11618)

Fig. 7 from [Rehemtulla et al. 2024](https://arxiv.org/abs/2401.15167)

## Citing `BTSbot`

If you use or reference `BTSbot` please cite [Rehemtulla et al. 2024](https://arxiv.org/abs/2401.15167) ([ADS](https://ui.adsabs.harvard.edu/abs/2024arXiv240115167R/abstract))
