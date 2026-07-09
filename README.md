# PyAiNetwork

PyAiNetork - this is a lightweight library to create small to medium-sized LM's.

---

## Main functions

- Creating neural networks
- neural network training
- learning and counting through numpy tables

## Installation

```bash
pip install PyAiNetwork
```

---

## examples of use normal

```bash
from PyAiNetwork import Network

net = Network(
    4, #input 
    1, #layers
    1, #neorons on layer
    4 #output
    'gelu' #activition function
)

input = [0.2,0.4,0.6,0.8]

output = net.forward(input)
print(output)

for i in range(1000):
    net.train(
        input, #input
        input, #output
        0.01 #learing rate
    )

output = net.forward(input)
print(output)
```
## examples of use profi

```bash
from PyAiNetwork import ProfNetwork

layers_neorons = [1]

net = ProfNetwork(
    4, #input 
    layers_neorons, #layers
    4 #output
    'gelu' #activition function
    'matrix' # math type (or every)
)

input = [0.2,0.4,0.6,0.8]

output = net.forward(input)
print(output)

for i in range(1000):
    net.train(
        input, #input
        input, #output
        0.01 #learing rate
    )

output = net.forward(input)
print(output)
```

# GitHub

https://github.com/eyes-studio/PyAiNetwork

---
