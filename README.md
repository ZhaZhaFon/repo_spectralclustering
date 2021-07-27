# Spectral Clustering
[![Python application](https://github.com/wq2012/SpectralCluster/workflows/Python%20application/badge.svg)](https://github.com/wq2012/SpectralCluster/actions) [![PyPI Version](https://img.shields.io/pypi/v/spectralcluster.svg)](https://pypi.python.org/pypi/spectralcluster) [![Python Versions](https://img.shields.io/pypi/pyversions/spectralcluster.svg)](https://pypi.org/project/spectralcluster) [![Downloads](https://pepy.tech/badge/spectralcluster)](https://pepy.tech/project/spectralcluster) [![codecov](https://codecov.io/gh/wq2012/SpectralCluster/branch/master/graph/badge.svg)](https://codecov.io/gh/wq2012/SpectralCluster) [![Documentation](https://img.shields.io/badge/api-documentation-blue.svg)](https://wq2012.github.io/SpectralCluster)

## Note

We are currently adding new functionalities to this library to include
some algorithms to appear in an upcoming paper. We are updating the APIs as
well. If you depend on our old API, please use an older version of this library.

## Overview

This is a Python re-implementation of the spectral clustering algorithm in the
paper [Speaker Diarization with LSTM](https://google.github.io/speaker-id/publications/LstmDiarization/).

![refinement](https://raw.githubusercontent.com/wq2012/SpectralCluster/master/resources/refinement.png)

## Disclaimer

**This is not a Google product.**

**This is not the original C++ implementation used by the paper.**

## Dependencies

* numpy
* scipy
* scikit-learn

## Installation

Install the [package](https://pypi.org/project/spectralcluster/) by:

```bash
pip3 install spectralcluster
```

or

```bash
python3 -m pip install spectralcluster
```

## Tutorial

Simply use the `predict()` method of class `SpectralClusterer` to perform
spectral clustering:

```python
from spectralcluster import SpectralClusterer

refinement_options = RefinementOptions(
  gaussian_blur_sigma=1,
  p_percentile=0.95,
  thresholding_soft_multiplier=0.01,
  thresholding_with_row_max=True)

clusterer = SpectralClusterer(
  min_clusters=2,
  max_clusters=100,
  refinement_options=refinement_options)

labels = clusterer.predict(X)
```

The input `X` is a numpy array of shape `(n_samples, n_features)`,
and the returned `labels` is a numpy array of shape `(n_samples,)`.

For the complete list of parameters of the clusterer, see
`spectralcluster/spectral_clusterer.py`.

For the complete list of refinement options, see
`spectralcluster/refinement.py`.

[![youtube_screenshot](resources/youtube_screenshot.jpg)](https://youtu.be/pjxGPZQeeO4)

## Citations

Our paper is cited as:

```
@inproceedings{wang2018speaker,
  title={Speaker diarization with lstm},
  author={Wang, Quan and Downey, Carlton and Wan, Li and Mansfield, Philip Andrew and Moreno, Ignacio Lopz},
  booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5239--5243},
  year={2018},
  organization={IEEE}
}
```

## FAQs

### Laplacian matrix

**Question:** Why are you performing eigen-decomposition directly on the affinity matrix instead of its Laplacian matrix? ([source](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053122))

**Answer:** No, we are not performing eigen-decomposition directly on the affinity matrix. In the sequence of refinement operations, the first operation is `CropDiagonal`, which replaces each diagonal element of the affinity matrix by the max non-diagonal value of the row. After this operation, the matrix has similar properties to a standard Laplacian matrix, and it is also less sensitive (thus more robust) to the Gaussian blur operation than a standard Laplacian matrix.

In the new version of this library, we support different types of Laplacian matrix now, including:

* None Laplacian (affinity matrix): W
* Unnormalized Laplacian: L = D - W
* Graph cut Laplacian: L' = D^{-1/2} L D^{-1/2}
* Random walk Laplacian: L' = D^{-1} L

### Cosine vs. Euclidean distance

**Question:** Your paper says the K-Means should be based on Cosine distance, but this repository is using Euclidean distance. Do you have a Cosine distance version?

**Answer:** We support Cosine distance now! Just set `custom_dist="cosine"` when initializing your `SpectralClusterer` object.

## Misc

Our new speaker diarization systems are now fully supervised, powered by
[uis-rnn](https://github.com/google/uis-rnn).
Check this [Google AI Blog](https://ai.googleblog.com/2018/11/accurate-online-speaker-diarization.html).

To learn more about speaker diarization, here is a curated list of resources:
[awesome-diarization](https://github.com/wq2012/awesome-diarization).
