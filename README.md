# Fractional Kolmogorov-Arnold Network (fKAN)

Fractional Kolmogorov-Arnold Network (fKAN) is a novel neural network that incorporates the distinctive attributes of Kolmogorov-Arnold Networks (KANs) with a trainable adaptive fractional-orthogonal Jacobi function as its basis function. This method offers several advantages, including non-polynomial behavior, activity for both positive and negative input values, faster execution, and better accuracy.

## Installation

To install fKAN, use the following command:

```bash
$ pip install fkan
```

## Example Usage

The current implementation of fKAN works with both the TensorFlow and PyTorch APIs.

### TensorFlow

```python
from tensorflow import keras
from tensorflow.keras import layers
from fkan.tensorflow import FractionalJacobiNeuralBlock as fJNB

model = keras.Sequential(
    [
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3)),
        fJNB(3),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(16),
        fJNB(2),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

```

### PyTorch

```python
import torch.nn as nn
from fkan.torch import FractionalJacobiNeuralBlock as fJNB

model = nn.Sequential(
    nn.Linear(1, 16),
    fJNB(3),
    nn.Linear(16, 32),
    fJNB(6),
    nn.Linear(32, 1),
)

```

## Experiments

The `example` folder contains the implementation of the experiments from the paper using fKAN. These experiments include:

### Deep Learning Tasks

- **Synthetic Regression**
- **MNIST Classification**
- **Fashion MNIST Image Denoising**
- **IMDB Sentiment Analysis**

### Physics Informed Deep Learning

- **Lane Emden Ordinary Differential Equation**
- **Burgers Partial Differential Equation**
- **Fractional Delay Differential Equation** with Caputo definition


## Current Limitations

-   Maximum allowed Jacobi polynomial degree is set to six.
- The current library is not compatible with other deep learning frameworks, but it can be converted easily.

## Contribution

We encourage the community to contribute by opening issues and submitting pull requests to help address these limitations and improve the overall functionality of fKAN.

## Contact

If you have any questions or encounter any issues, please open an issue in this repository (preferred) or reach out to the author directly.

## Citation

If you use fKAN in your research, please cite our paper:

```
@misc{aghaei2024fkan,
      title={fKAN: Fractional Kolmogorov-Arnold Networks with trainable Jacobi basis functions},
      author={Alireza Afzal Aghaei},
      year={2024},
      eprint={2406.07456},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
