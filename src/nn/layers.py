import numpy as np
from initialization import initial_weight_bias
from activation import relu,relu_dif,sigmoid,sigmoid_def,tanh,tanh_def
class  Linear:
    def __init__(self,input_neuron,output_neuron,random_state) -> None:
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.random_state = random_state
        self.w,self.b = initial_weight_bias(input_neuron,output_neuron,random_state)
    def forward(self,x):
        z = self.w @ x + self.b
        return z
    def backward(self,x,dz):
        dw = np.outer(dz,x)
        db = dz
        dx = self.w.T @ dz
        return dw , db,dx

class Relu:
    def forward(self,z):
        a = relu(z)
        return a
    def backward(self, z, dL):
        da = relu_dif(z)*dL
        return da

class Sigmoid:
    def forward(self,z):
        a = sigmoid(z)
        return a
    def backward(self, z, dL):
        da = sigmoid_def(z)*dL
        return da

class Tanh:
    def forward(self,z):
        a = tanh(z)
        return a
    def backward(self, z, dL):
        da = tanh_def(z)*dL
        return da
