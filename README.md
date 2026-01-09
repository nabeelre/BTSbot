<p align="center">
  <img
    src="https://github.com/nabeelre/BTSbot/assets/33795390/1b8586b1-5a89-4b84-a971-bf85fe722696"
    alt="BTSbot Logo"
    width="250px"
  />
</p>

[![arXiv](https://img.shields.io/badge/Original%20Publication-2401.15167-b31b1b.svg)](https://iopscience.iop.org/article/10.3847/1538-4357/ad5666)
[![arXiv](https://img.shields.io/badge/Architecture%20Benchmarking-2512.11957-b31b1b.svg)](https://arxiv.org/abs/2512.11957)
[![arXiv](https://img.shields.io/badge/ICML%20Paper-2307.07618-b31b1b.svg)](https://arxiv.org/abs/2307.07618)

`BTSbot` is a multi-modal deep vision model for automating supernova identification and follow-up in the [Zwicky Transient Facility (ZTF)](https://www.ztf.caltech.edu) [Bright Transient Survey (BTS)](https://sites.astro.caltech.edu/ztf/bts/bts.php). 

`BTSbot` contributed to the first supernova to be fully automatically discovered, confirmed, classified, and shared ([AstroNote](https://www.wis-tns.org/astronotes/astronote/2023-265), [press release](https://news.northwestern.edu/stories/2023/10/first-supernova-detected-confirmed-classified-and-shared-by-ai/)) as well automated space-based supernova follow-up ([AstroNote](https://www.wis-tns.org/astronotes/astronote/2025-209)). See this [animated walkthrough](https://www.youtube.com/watch?v=qUwlQflDdEo) of the fully-automated BTS workflow.

Presented at the [ML for Astrophysics workshop](https://ml4astro.github.io/icml2023/) at [ICML 2023](https://icml.cc/Conferences/2023) ([Extended abstract](https://arxiv.org/abs/2307.07618))

The training set for the original production model is available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10839690).


## Installation

Install with pip:

```bash
pip install btsbot
```

## Usage

`BTSbot` models are now available on the [HuggingFace Hub](https://huggingface.co/collections/nabeelr/pre-trained-off-the-shelf-btsbots) and can be downloaded and loaded into Python automatically. Here's how simple it is to get started:

```python
import btsbot

# This will automatically download the model if not already present locally
model = btsbot.load_HF_model(
    architecture="convnext",  # or "maxvit"
    multi_modal=True,         # Set to False for image-only models
    pretrain="galaxyzoo"      # or "imagenet", "randinit"
)
```

The `model` object can then be used for inference. For a complete inference example, see `inference_example.py` which demonstrates:
- Loading example data (triplets and metadata)
- Running inference on batches
- Processing predictions

## Citing `BTSbot`

If you use or reference `BTSbot` please cite [Rehemtulla et al. 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad5666) ([ADS](https://ui.adsabs.harvard.edu/abs/2024ApJ...972....7R/abstract)).
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

If you use or reference a pre-trained `BTSbot` model like the updated ConvNeXt-based `BTSbot`
or any `BTSbot` output from 2026 or beyond, please also cite [our follow-up publication](https://arxiv.org/abs/2512.11957).
```
@ARTICLE{Rehemtulla+2025,
      title={Pre-training vision models for the classification of alerts from wide-field time-domain surveys}, 
      author={Nabeel Rehemtulla and Adam A. Miller and Mike Walmsley and Ved G. Shah and Theophile Jegou du Laz and Michael W. Coughlin and Argyro Sasli and Joshua Bloom and Christoffer Fremling and Matthew J. Graham and Steven L. Groom and David Hale and Ashish A. Mahabal and Daniel A. Perley and Josiah Purdum and Ben Rusholme and Jesper Sollerman and Mansi M. Kasliwal},
      year={2025},
      eprint={2512.11957},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2512.11957}, 
}
```
