import numpy as np
def initial_weight_bias(input_neuron,output_neuron,randon_state):
    np.random.seed(randon_state)
    w = np.random.randn(output_neuron,input_neuron)
    b = np.zeros((output_neuron,))
    return w,b
