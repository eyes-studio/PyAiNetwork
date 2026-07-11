import numpy as np
import json
import random
import math

class MathEngine:
    @staticmethod
    def sigmoid(x):
        if isinstance(x, np.ndarray):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        else:
            if x < -20:
                return 0
            if x > 20:
                return 1

            return 1.0/(1.0+math.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = MathEngine.sigmoid(x)
        return s * (1.0 - s)
    
    @staticmethod
    def relu(x):
        if isinstance(x, np.ndarray):
            return np.maximum(0.0, x)
        else:
            return max(0.0, x)
    
    @staticmethod
    def relu_derivative(x):
        if isinstance(x, np.ndarray):
            return np.where(x > 0, 1.0, 0.0)
        else:
            return 1.0 if x > 0 else 0.0
    
    @staticmethod
    def gelu(x):
        if isinstance(x, np.ndarray):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        else:
            return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

    
    @staticmethod
    def gelu_derivative(x):
        c = np.sqrt(2 / np.pi) if isinstance(x, np.ndarray) else math.sqrt(2 / math.pi)
        tanh = np.tanh(c * (x + 0.044715 * x**3)) if isinstance(x, np.ndarray) else math.tanh(c * (x + 0.044715 * x**3))
        sech = 1 - tanh**2
        return 0.5 * (1 + tanh + x * sech * c * (1 + 3 * 0.044715 * x**2))

class Neoron:
    def __init__(self,inputs,activition='gelu'):
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
                answer.append(inputs[i] * self.multiplier[i])
    
            self.last_sum = sum(answer) + self.addition
            output = MathEngine.sigmoid(self.last_sum)
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

class Neoron_matrix:
    def __init__(self, inputs, activation='gelu'):
        self.a = activation
        
        # Ваги як NumPy вектор, не список!
        self.multiplier = np.random.uniform(-0.5, 0.5, inputs)
        self.addition = 0
    
    def forward(self, inputs):
        self.last_input = np.array(inputs) if not isinstance(inputs, np.ndarray) else inputs
        
        # Матричне множення замість циклу!
        self.last_sum = np.dot(self.last_input, self.multiplier) + self.addition
        
        if self.a == 'gelu':
            output = MathEngine.gelu(self.last_sum)
        elif self.a == 'relu':
            output = MathEngine.relu(self.last_sum)
        elif self.a == 'sigmoid':
            output = MathEngine.sigmoid(self.last_sum)
        else:
            print("activation name error")
        
        return output
    
    def train(self, inputs, output, epox, learning_rate=0.01):
        for _ in range(epox):
            prediction = self.forward(inputs)
            error = output - prediction
            
            # Матричне оновлення замість циклу!
            self.multiplier += error * self.last_input * learning_rate
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

        error = error * gradient
        
        # Матричне оновлення
        self.multiplier += error * self.last_input * learning_rate
        self.addition += error * learning_rate


class Layer:
    def __init__(self,input,neporons, activition='gelu', counting_system='every'):
        if counting_system == 'every':
            self.matric = False
        elif counting_system == 'matrix':
            self.matric = True
        else:
            print("EROR")
        
        self.a = activition

        self.n = []
        if self.matric == False:
            for i in range(neporons):
                self.n.append(Neoron(input, activition))
        else:
            for i in range(neporons):
                self.n.append(Neoron_matrix(input, activition))

    def forward(self,inputs):
        if self.matric == False:
            output = []

            for neuron in self.n:
                output.append(neuron.forward(inputs))

            return output
        else:
            output = []
            for neuron in self.n:
                output.append(neuron.forward(inputs))
            for i in range(len(output)):
                if isinstance(output[i], np.generic):
                    output[i] = output[i].item()

            return output
    
    def train(self, inputs, outputs, epox, learning_rate=0.01):
        if self.matric == False:
            for _ in range(epox):
                for neuron, output in zip(self.n, outputs):
                    neuron.train(inputs, output, 1, learning_rate)
        else:
            for _ in range(epox):
                for neuron, output in zip(self.n, outputs):
                    neuron.train(inputs, output, 1, learning_rate)

    def backward(self, errors, learning_rate):
        if self.matric == False:
            new_errors = [0] * len(self.n[0].multiplier)

            for neuron, error in zip(self.n, errors):
                neuron.backward(error, learning_rate)
        
                for i in range(len(neuron.multiplier)):
                    new_errors[i] += error * neuron.multiplier[i]

            return new_errors
        else:
            weights = np.array([n.multiplier for n in self.n])
            errors_array = np.array(errors)
            new_errors = np.dot(weights.T, errors_array)
            
            for neuron, error in zip(self.n, errors):
                neuron.backward(error, learning_rate)
            
            return new_errors.tolist()
    
class Network:
    def __init__(self,input_neorons,layers,neorons_on_layer,output_neoron, activition='gelu'):
        self.a = activition
        self.ni = Layer(input_neorons,input_neorons, activition)

        self.l = []
        for i in range(layers):
            if i == 0:
                self.l.append(Layer(input_neorons, neorons_on_layer, activition))

            else:
                self.l.append(Layer(neorons_on_layer, neorons_on_layer, activition))
        
        self.no = Layer(neorons_on_layer,output_neoron, activition)

    def forward(self, inputs):
        neorons_answer = self.ni.forward(inputs)

        lan = len(self.l)
        for i in range(lan):
            neorons_answer = self.l[i].forward(neorons_answer)

        neorons_answer = self.no.forward(neorons_answer)

        return neorons_answer
    
    def train(self, input, output, learning_rate=0.01):

        prediction = self.forward(input)


        errors = []

        for i in range(len(output)):
            errors.append(output[i] - prediction[i])


        errors = self.no.backward(
            errors,
            learning_rate
        )


        for i in range(len(self.l)-1, -1, -1):
            errors = self.l[i].backward(
                errors,
                learning_rate
            )

        errors = self.ni.backward(errors, learning_rate)

class ProfNetwork:
    def __init__(self,input_neorons,layers_neorons,output_neorons, activition='gelu', counting_system='every'):
        if counting_system == 'every':
            self.matric = False
        elif counting_system == 'matrix':
            self.matric = True
        else:
            print("EROR")


        self.ni = Layer(input_neorons, input_neorons, activition, counting_system)

        self.l = []
        neoron_inp = input_neorons
    
        for neurons in layers_neorons:
            self.l.append(
                Layer(neoron_inp, neurons, activition, counting_system)
            )
            neoron_inp = neurons

        self.no = Layer(neoron_inp, output_neorons, activition, counting_system)

    def forward(self, inputs):
        if self.matric == False:
            neorons_answer = self.ni.forward(inputs)

            lan = len(self.l)
            for i in range(lan):
                neorons_answer = self.l[i].forward(neorons_answer)

            neorons_answer = self.no.forward(neorons_answer)

            return neorons_answer
        else:
            neorons_answer = self.ni.forward(inputs)

            lan = len(self.l)
            for i in range(lan):
                neorons_answer = self.l[i].forward(neorons_answer)

            neorons_answer = self.no.forward(neorons_answer)

            return neorons_answer
    
    def train(self, input, output, learning_rate=0.01):
        if self.matric == False:

            prediction = self.forward(input)


            errors = []

            for i in range(len(output)):
                errors.append(output[i] - prediction[i])


            errors = self.no.backward(
                errors,
                learning_rate
            )


            for i in range(len(self.l)-1, -1, -1):
                errors = self.l[i].backward(
                     errors,
                     learning_rate
                )

            self.ni.backward(
                errors,
                learning_rate
            )
        else:
            prediction = self.forward(input)

            errors = []
            for i in range(len(output)):
                errors.append(output[i] - prediction[i])

            errors = self.no.backward(errors, learning_rate)

            for i in range(len(self.l)-1, -1, -1):
                errors = self.l[i].backward(errors, learning_rate)

            self.ni.backward(errors, learning_rate)
        
    def save(self, json_file):
        data = {
            'ni': [],
            'l': [],
            'no': []
        }
        
        # Зберігаємо input layer
        for neuron in self.ni.n:
            data['ni'].append({
                'multiplier': neuron.multiplier if isinstance(neuron.multiplier, list) else neuron.multiplier.tolist(),
                'addition': float(neuron.addition)
            })
        
        # Зберігаємо приховані слої
        for layer in self.l:
            layer_data = []
            for neuron in layer.n:
                layer_data.append({
                    'multiplier': neuron.multiplier if isinstance(neuron.multiplier, list) else neuron.multiplier.tolist(),
                    'addition': float(neuron.addition)
                })
            data['l'].append(layer_data)
        
        # Зберігаємо output layer
        for neuron in self.no.n:
            data['no'].append({
                'multiplier': neuron.multiplier if isinstance(neuron.multiplier, list) else neuron.multiplier.tolist(),
                'addition': float(neuron.addition)
            })
        
        with open(json_file, 'w') as f:
            json.dump(data, f)

    def load(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Завантажуємо input layer
        for i, neuron_data in enumerate(data['ni']):
            if isinstance(self.ni.n[i].multiplier, list):
                self.ni.n[i].multiplier = neuron_data['multiplier']
            else:
                self.ni.n[i].multiplier = np.array(neuron_data['multiplier'])
            self.ni.n[i].addition = neuron_data['addition']
        
        # Завантажуємо приховані слої
        for layer_idx, layer_data in enumerate(data['l']):
            for neuron_idx, neuron_data in enumerate(layer_data):
                if isinstance(self.l[layer_idx].n[neuron_idx].multiplier, list):
                    self.l[layer_idx].n[neuron_idx].multiplier = neuron_data['multiplier']
                else:
                    self.l[layer_idx].n[neuron_idx].multiplier = np.array(neuron_data['multiplier'])
                self.l[layer_idx].n[neuron_idx].addition = neuron_data['addition']
        
        # Завантажуємо output layer
        for i, neuron_data in enumerate(data['no']):
            if isinstance(self.no.n[i].multiplier, list):
                self.no.n[i].multiplier = neuron_data['multiplier']
            else:
                self.no.n[i].multiplier = np.array(neuron_data['multiplier'])
            self.no.n[i].addition = neuron_data['addition']

