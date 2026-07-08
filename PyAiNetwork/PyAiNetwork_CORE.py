import math
import random
from erormanager import EROR

class MathEngine:
    def sigmoid(x):
        if x < -20:
            return 0
        if x > 20:
            return 1

        return 1.0/(1.0+math.exp(-x))
    
    def sigmoid_derivative(x):
        s = MathEngine.sigmoid(x)
        return s * (1.0 - s)
    
    def relu(x):
        return max(0.0, x)
    
    def relu_derivative(x):
        return 1.0 if x > 0 else 0.0
    
    def gelu(x):
        return 0.5 * x * (
            1 + math.tanh(math.sqrt(2 / math.pi) *(x + 0.044715 * x**3))
        )
    
    def gelu_derivative(x):
        c = math.sqrt(2 / math.pi)

        tanh = math.tanh(
            c * (x + 0.044715 * x**3)
        )

        sech = 1 - tanh**2

        return 0.5 * (
            1 + tanh +
            x * sech * c * (1 + 3 * 0.044715 * x**2)
        )

class Neoron:
    def __init__(self,inputs,activition='gelu', counting_system='every'):
        if counting_system == 'every':
            self.matric = False
        elif counting_system == 'matrix':
            self.matric = True
        else:
            EROR("IM", counting_system)
        self.a = activition

        self.multiplier = []
        for i in range(inputs):
            self.multiplier.append(random.uniform(-0.5,0.5))
        
        self.addition = 0
    
    def forward(self,inputs):
        rang = len(inputs)

        self.last_input = inputs

        if self.a == 'gelu':
            answer = []
            for i in range(rang):
                answer.append(inputs[i] * self.multiplier[i])
    
            self.last_sum = sum(answer) + self.addition
            output = MathEngine.gelu(self.last_sum)
            return output
        
        elif self.a == 'relu':
            answer = []
            for i in range(rang):
                answer.append(inputs[i] * self.multiplier[i])
    
            self.last_sum = sum(answer) + self.addition
            output = MathEngine.relu(self.last_sum)
            return output

        elif self.a == 'sigmoid':
            answer = []

            for i in range(rang):
                answer.append(MathEngine.sigmoid(inputs[i]) * self.multiplier[i])
            
            output = sum(answer) + self.addition
            self.last_sum = output
            return output
        else:
            print("activition name eror")

    def train(self, inputs, output, epox, learning_rate=0.01):
        for _ in range(epox):
            prediction = self.forward(inputs)

            error = output - prediction

            for i in range(len(self.multiplier)):
                self.multiplier[i] += error * inputs[i] * learning_rate

            self.addition += error * learning_rate

        error = output - prediction

        for i in range(len(self.multiplier)):
            self.multiplier[i] += error * inputs[i] * learning_rate

        self.addition += error * learning_rate

    def backward(self, error, learning_rate):
        if self.a == "gelu":
            gradient = MathEngine.gelu_derivative(self.last_sum)

        elif self.a == "relu":
            gradient = MathEngine.relu_derivative(self.last_sum)

        elif self.a == "sigmoid":
            gradient = MathEngine.sigmoid_derivative(self.last_sum)

        else:
            gradient = 1


        error *= gradient


        for i in range(len(self.multiplier)):
            self.multiplier[i] += error * self.last_input[i] * learning_rate

        self.addition += error * learning_rate