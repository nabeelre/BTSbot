<p align="center">
  <img
    src="https://github.com/nabeelre/BTSbot/assets/33795390/1b8586b1-5a89-4b84-a971-bf85fe722696"
    alt="BTSbot Logo"
    width="250px"
  />
</p>

[![arXiv](https://img.shields.io/badge/Original%20Publication-2401.15167-b31b1b.svg)](https://iopscience.iop.org/article/10.3847/1538-4357/ad5666)
[![arXiv](https://img.shields.io/badge/Architecture%20Benchmarking-2512.XXXXX-b31b1b.svg)]()
[![arXiv](https://img.shields.io/badge/ICML%20Paper-2307.07618-b31b1b.svg)](https://arxiv.org/abs/2307.07618)

`BTSbot` is a multi-modal convolutional neural network for automating supernova identification and follow-up in the [Zwicky Transient Facility (ZTF)](https://www.ztf.caltech.edu) [Bright Transient Survey (BTS)](https://sites.astro.caltech.edu/ztf/bts/bts.php). 

`BTSbot` contributed to the first supernova to be fully automatically discovered, confirmed, classified, and shared ([AstroNote](https://www.wis-tns.org/astronotes/astronote/2023-265), [press release](https://news.northwestern.edu/stories/2023/10/first-supernova-detected-confirmed-classified-and-shared-by-ai/)) as well automated space-based supernova follow-up ([AstroNote](https://www.wis-tns.org/astronotes/astronote/2025-209)).

Presented at the [ML for Astrophysics workshop](https://ml4astro.github.io/icml2023/) at [ICML 2023](https://icml.cc/Conferences/2023) ([Extended abstract](https://arxiv.org/abs/2307.07618))

The training set for the original production model is available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10839690).

Also see this [animated walkthrough](https://www.youtube.com/watch?v=qUwlQflDdEo) of the fully-automated BTS workflow

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

## Citing `BTSbot`

If you use or reference `BTSbot` please cite [Rehemtulla et al. 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad5666) ([ADS](https://ui.adsabs.harvard.edu/abs/2024ApJ...972....7R/abstract)).

BibTeX entry for the `BTSbot` paper:
```
@ARTICLE{Rehemtulla+2024,
       author = {{Rehemtulla}, Nabeel and {Miller}, Adam A. and {Jegou Du Laz}, Theophile and {Coughlin}, Michael W. and {Fremling}, Christoffer and {Perley}, Daniel A. and {Qin}, Yu-Jing and {Sollerman}, Jesper and {Mahabal}, Ashish A. and {Laher}, Russ R. and {Riddle}, Reed and {Rusholme}, Ben and {Kulkarni}, Shrinivas R.},
        title = "{The Zwicky Transient Facility Bright Transient Survey. III. BTSbot: Automated Identification and Follow-up of Bright Transients with Deep Learning}",
      journal = {\apj},
     keywords = {Time domain astronomy, Sky surveys, Supernovae, Convolutional neural networks, 2109, 1464, 1668, 1938, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = sep,
       volume = {972},
       number = {1},
          eid = {7},
        pages = {7},
          doi = {10.3847/1538-4357/ad5666},
archivePrefix = {arXiv},
       eprint = {2401.15167},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...972....7R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

If you use or reference the updated ConvNeXt-based based `BTSbot` model, please also cite [Rehemtulla et al. 2025]() ([ADS]()).

BibTeX entry for the follow-up `BTSbot` study:
```

```
