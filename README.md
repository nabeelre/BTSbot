# `BTSbot`

`BTSbot` is a machine learning tool to automate source identification and follow-up for the [Zwicky Transient Facility (ZTF)](https://www.ztf.caltech.edu) [Bright Transient Survey (BTS)](https://sites.astro.caltech.edu/ztf/bts/bts.php). ZTF is a robotic observatory that looks for changes in the night sky by repeatedly taking new images and comparing them with historical ones. ZTF finds thousands of supernovae (SNe; the explosive deaths of stars) each year; BTS endeavors to collect follow-up observations of to classify them and build a very large, complete sample of SNe. `BTSbot` automatically looks through ZTF data to find the SNe of interest to BTS and send follow-up observation requests for them. [Fritz](https://github.com/fritz-marshal/fritz)/[SkyPortal](https://github.com/skyportal/skyportal) and [Kowalski](https://github.com/skyportal/kowalski) aid in coordinating between ZTF's data stream and `BTSbot`

`BTSbot` was presented at the [ML for Astrophysics workshop](https://ml4astro.github.io/icml2023/) at [ICML 2023](https://icml.cc/Conferences/2023) ([Extended abstract](https://arxiv.org/abs/2307.07618))



## `BTSbot: a multi-modal convolutional neural network`



## Performance

`BTSbot` finds nearly all SNe of interest from the input data stream (~100% completeness) with little contamination from uninteresting phenomena (93% purity) and does so as quick as humans typically do.



## Usage



#### Citing `BTSbot`

If you use or reference `BTSbot` please cite [Rehemtulla et al. 2023](https://arxiv.org/abs/2307.07618)

Rehemtulla et al. 2024 (in prep)



## Requirements

