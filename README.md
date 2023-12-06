# `BTSbot`

`BTSbot` is a machine learning tool to automate source identification and follow-up for the [Zwicky Transient Facility (ZTF)](https://www.ztf.caltech.edu) [Bright Transient Survey (BTS)](https://sites.astro.caltech.edu/ztf/bts/bts.php). ZTF is a robotic observatory that looks for changes in the night sky by repeatedly taking new images and comparing them with historical ones. ZTF finds thousands of supernovae (SNe; the explosive deaths of stars) each year, and BTS endeavors to classify them with follow-up observations and build a large, complete sample of SNe. `BTSbot` automatically looks through ZTF data to find the SNe of interest to BTS and send follow-up observation requests for them. [Fritz](https://github.com/fritz-marshal/fritz)/[SkyPortal](https://github.com/skyportal/skyportal) and [Kowalski](https://github.com/skyportal/kowalski) aid in coordinating between ZTF's data stream and `BTSbot`

Presented at the [ML for Astrophysics workshop](https://ml4astro.github.io/icml2023/) at [ICML 2023](https://icml.cc/Conferences/2023) ([Extended abstract](https://arxiv.org/abs/2307.07618))

`BTSbot` contributed to the first SN to be fully automatically discovered, confirmed, classified, and shared. ([AstroNote](https://www.wis-tns.org/astronotes/astronote/2023-265), [press release](https://news.northwestern.edu/stories/2023/10/first-supernova-detected-confirmed-classified-and-shared-by-ai/))

## A multi-modal convolutional neural network

![model](https://github.com/nabeelre/BTSbot/assets/33795390/c33431eb-2a0d-4ed1-8b30-11a5810699c4)

Fig. 5 from Rehemtulla et al. 2024 (in prep)

## Usage

## Performance

`BTSbot` finds nearly all SNe of interest from the input data stream (~100% completeness) with little contamination from uninteresting phenomena (93% purity) and does so as quickly as humans typically do.

![test_performance](https://github.com/nabeelre/BTSbot/assets/33795390/c9d1b930-c980-4e59-95d1-074bf7e11618)

Fig. 7 from Rehemtulla et al. 2024 (in prep)

## Citing `BTSbot`

If you use or reference `BTSbot` please cite [Rehemtulla et al. 2023](https://arxiv.org/abs/2307.07618) ([ADS](https://ui.adsabs.harvard.edu/abs/2023arXiv230707618R/abstract))

Rehemtulla et al. 2024 (in prep)
