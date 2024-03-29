# BatchED
ECCV22 Paper "[Batch-efficient EigenDecomposition for Small and Medium Matrices](https://arxiv.org/abs/2207.04228)"

<img src="numerical_test_batch.jpg" width="89%">

We implement a Pytorch-based batch-efficient ED solver for small and medium matrices (dim<32), which is dedicated to the application scenarios of computer vision. The core part of the algorithm is based on the QR iteration with Double Wilkinson shifts and some other acceleration techniques carefully designed for the best batch efficiency. **Our Pytorch-implemented solver performs the ED entirely via batched matrix-matrix multiplication, which processes all the matrices simultaneously and thus fully exploits the parallel computational power of GPUs**.

<img src="num_speed.jpg" width="89%">

We have also implemented a Divide-and-Conquer based batch-efficient ED solver for relatively larger matrices (dim<64). Please refer to [the document](https://github.com/KingJamesSong/BatchED/blob/main/BatchEfficient_ED.pdf) and [the code](https://github.com/KingJamesSong/BatchED/blob/main/utils_ed_plus.py) for the algorithm and implementation, respectively. 

## Usage 

Download `utils_ed.py` to your project folder and add the folowing lines to your main code.

```python
# Import batch ed
from utils_ed import Batched_ED #QR-based ED for dim<32
from utils_ed_plus import Bacthed_ED_Plus #DC-based ED for dim<64

batched_ed = Batched_ED.apply
batched_ed_plus = Bacthed_ED_Plus.apply

# Run batch ed for a matrix
eigen_vectors,eigen_values = batched_ed(cov)
eigen_vectors2,eigen_values2 = batched_ed_plus(cov)
```

The complete exemplery usage and comparison is given in `main.py`. For batched matrices of size `512x4x4`, the output log is:

```python
SVD Time 0.46
Batched ED (QR) Time 0.012
Batched ED Plus (DC) Time 0.017
```

For batched matrices of size `512x32x32`, the log is:
```python
SVD Time 0.59
Batched ED (QR) Time 0.36
Batched ED Plus (DC) Time 0.22
```


## Requirements

`torch<=1.7.1` and install `mpmath` if you need the ED gradients. 

## Computer Vision Experiments

Please refer to [Fast Differentiable Matrix Square Root](https://github.com/KingJamesSong/FastDifferentiableMatSqrt) for all the real-world computer vision experiments. We want to maintain this repository as a standalone lib for batch ED. So it should be as clean as possible.

## Citation

Please consider citing our paper if you think the code is helpful to your research.

```
@inproceedings{song2022batch,
  title={Batch-efficient EigenDecomposition for Small and Medium Matrices},
  author={Song, Yue and Sebe, Nicu and Wang, Wei},
  booktitle={ECCV},
  year={2022}
}
```

## Contact

If you have any questions or suggestions, please feel free to contact me. Alternatively, if you are interested in implementing a CUDA version, please drop me an e-mail and create a seperate branch.

`yue.song@unitn.it`

