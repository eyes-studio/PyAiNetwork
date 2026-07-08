# PyAiNetwork

PyAiNetwork is an open-source Python library for creating and training neural networks with simple and beginner-friendly code.

Install:

```bash
pip install PyAiNetwork
```

---

# About

PyAiNetwork is designed to make neural network development simple.

Instead of writing hundreds of lines of code for neurons, layers, weights, and training, you can create a neural network with just a few commands.

The library is suitable for:

- Learning how neural networks work
- Creating simple AI projects
- Experimenting with custom network architectures
- Building your own AI systems

---

# Developers

**Eyes Studio**

Eyes Studio is an independent software developer focused on AI tools and open-source projects.

---

# Features

- ✅ Simple neural network API
- ✅ Multiple hidden layers
- ✅ GELU activation
- ✅ ReLU activation
- ✅ Sigmoid activation
- ✅ Built-in training
- ✅ Lightweight implementation
- ✅ Pure Python
- ✅ Open Source

---

# Installation

Install from PyPI:

```bash
pip install PyAiNetwork
```

Import:

```python
from PyAiNetwork import Network
```

---

# Quick Start

Create your first neural network.

```python
from PyAiNetwork import Network

net = Network(
    2,  # input neurons
    2,  # hidden layers
    4,  # neurons in every hidden layer
    1   # output neurons
)

result = net.forward([0.5, 1.0])

print(result)
```

---

# Training

Example:

```python
from PyAiNetwork import Network

net = Network(2,1,4,1)

for i in range(100):
    net.train(
        [1,0],
        [1]
    )

print(net.forward([1,0]))
```

---

# Activation Functions

PyAiNetwork currently supports:

```python
activition="gelu"
```

```python
activition="relu"
```

```python
activition="sigmoid"
```

Example:

```python
net = Network(
    2,
    2,
    8,
    1,
    activition="relu"
)
```

---

# Network

Simple neural network.

Constructor:

```python
Network(
    input_neorons,
    layers,
    neorons_on_layer,
    output_neorons,
    activition="gelu"
)
```

Parameters:

| Parameter | Description |
|-----------|-------------|
| input_neorons | Number of input neurons |
| layers | Number of hidden layers |
| neorons_on_layer | Neurons in every hidden layer |
| output_neorons | Number of output neurons |
| activition | Activation function |

---

# ProfNetwork

Advanced neural network.

Unlike `Network`, every hidden layer can have a different number of neurons.

Example:

```python
from PyAiNetwork import ProfNetwork

net = ProfNetwork(
    2,
    [8,16,8],
    1
)
```

Architecture:

```
2 → 8 → 16 → 8 → 1
```

Constructor:

```python
ProfNetwork(
    input_neorons,
    layers_neorons,
    output_neorons,
    activition="gelu"
)
```

Example:

```python
net = ProfNetwork(
    3,
    [32,64,64,32],
    5
)
```

---

# Roadmap

Future versions may include:

- Adam optimizer
- Model saving/loading
- Batch training
- More activation functions
- Loss functions
- Better performance
- More neural network types

---

# License

MIT License

Copyright (c) 2026 Eyes Studio
