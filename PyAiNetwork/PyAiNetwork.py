from PyAiNetwork_CORE import Neoron,MathEngine
from erormanager import EROR

class Layer:
    def __init__(self,input,neporons, activition='gelu'):
        self.a = activition

        self.n = []
        for i in range(neporons):
            self.n.append(Neoron(input, activition))

    def forward(self,inputs):
        output = []

        for neuron in self.n:
            output.append(neuron.forward(inputs))

        return output
    
    def train(self, inputs, outputs, epox, learning_rate=0.01):
        for _ in range(epox):
            for neuron, output in zip(self.n, outputs):
                neuron.train(inputs, output, 1, learning_rate)


    def backward_train(self, inputs, errors, learning_rate=0.01):
        for neuron, error in zip(self.n, errors):
            for i in range(len(neuron.multiplier)):
                neuron.multiplier[i] += error * inputs[i] * learning_rate

            neuron.addition += error * learning_rate

    def backward(self, errors, learning_rate):
        new_errors = []

        for neuron, error in zip(self.n, errors):
            new_errors.append(error)

            neuron.backward(error, learning_rate)

        return new_errors

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

class ProfNetwork:
    def __init__(self,input_neorons,layers_neorons,output_neorons, activition='gelu', counting_system='every'):
        if counting_system == 'every':
            self.matric = False
        elif counting_system == 'matrix':
            self.matric = True
        else:
            EROR("IM", "Sorry, but the matrices are temporarily not working.")


        self.ni = Layer(input_neorons, input_neorons, activition)

        self.l = []
        neoron_inp = input_neorons
    
        for neurons in layers_neorons:
            self.l.append(
                Layer(neoron_inp, neurons, activition)
            )
            neoron_inp = neurons

        self.no = Layer(neoron_inp, output_neorons, activition)

    def forward(self, inputs):
        if self.matric == False:
            neorons_answer = self.ni.forward(inputs)

            lan = len(self.l)
            for i in range(lan):
                neorons_answer = self.l[i].forward(neorons_answer)

            neorons_answer = self.no.forward(neorons_answer)

            return neorons_answer
        else:
            EROR("IM", "Sorry, but the matrices are temporarily not working.")
    
    def train(self, input, output, learning_rate=0.01):
        if self.matric == True:

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
            EROR("IM", "Sorry, but the matrices are temporarily not working.")

com = Network(2, 1, 1, 2)

inpy = [1,2]

output = com.forward(inpy)
print(output)

for i in range(1000):
    com.train(inpy,inpy)

output = com.forward(inpy)
print(output)