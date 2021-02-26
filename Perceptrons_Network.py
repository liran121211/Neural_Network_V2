from matplotlib import pyplot as plt
import numpy as np
from math import exp
from matplotlib.animation import FuncAnimation as FA
import time
import random as rnd

def multiplyMatrix(matrixA, matrixB):
    if ( isinstance(matrixA, (np.ndarray)) and isinstance(matrixB, (np.ndarray))):
        return np.multiply(matrixA, matrixB)
    else:
        return np.dot(matrixA, matrixB)

def mapMatrix(matrix,function):
    for rows in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            matrix[rows][column] = function(matrix[rows][column])
    return matrix

def sigmoid(n):
    return (1 / (1 + exp(-n) ) )

def derivativeSigmoid(y):
    #return sigmoid(n) * (1 - sigmoid(n))
    return y * (1 - y)

def setup(inputNodes = 2, hiddenNodes = 2, outputNodes = 1, iterations = 500,jumps = 1):
    NN = NeuralNetwork(inputNodes,hiddenNodes,outputNodes) #Init NeuralNetwork Object
    print('Initializing Neural Network ({0}X{1}X{2})'.format(inputNodes, hiddenNodes, outputNodes))
    time.sleep(1)
    print('Calculating Matrixes of DATA...')
    DATA = [ { 'inputs': [0, 1] , 'target': [1] },
            { 'inputs': [1, 0] , 'target': [1] },
            { 'inputs': [0, 0] , 'target': [0] },
            { 'inputs': [1, 1] , 'target': [0] }
            ]

    graphicArray = np.empty([iterations, 5])

    for row in range(iterations):
        randomize_item = rnd.randint(0, 3)
        for column in range(len(DATA)):
            NN.train(DATA[randomize_item]['inputs'],DATA[randomize_item]['target'])

        graphicArray[row, 0] = NN.prediction(np.array([0, 1]).reshape(2, 1))[0]
        graphicArray[row, 1] = NN.prediction(np.array([1, 0]).reshape(2, 1))[0]
        graphicArray[row, 2] = NN.prediction(np.array([1, 1]).reshape(2, 1))[0]
        graphicArray[row, 3] = NN.prediction(np.array([0, 0]).reshape(2, 1))[0]
        graphicArray[row, 4] = 1

    print('Done Filling MatPlotLib Array...\n Initiating Graphics...')
    print('Final Predictions: {0:.2f}|{1:.2f}|{2:.2f}|{3:.2f}'.format(graphicArray[iterations-1, 0],
                                                               graphicArray[iterations-1, 1],
                                                               graphicArray[iterations-1, 2],
                                                               graphicArray[iterations-1, 3]) )
    NN.draw(graphicArray,jumps)
    return NN, graphicArray

class NeuralNetwork():

    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr = 0.10):
        #Layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        #Weights
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.output_nodes, self.hidden_nodes))

        #Bias
        self.bias_hidden = np.random.uniform(-1, 1, (self.hidden_nodes, 1))
        self.bias_output = np.random.uniform(-1, 1, (self.output_nodes, 1))

        #Learning Rate
        self.learning_rate = lr

    def __str__(self):
        matrixes = [self.weights_input_hidden, self.weights_hidden_output]

        return (
                'Input Nodes: {0} | Hidden Nodes: {1} | Output Nodes: {2}\n\n'
                ' Weights_IH:\n {3} \n\n Weights_HO:\n {4} \n\n'
                ' Vector_H: \n{5} \n\n Vector_O: \n{6}'
                .format(self.input_nodes,
                        self.hidden_nodes,
                        self.output_nodes,
                        matrixes[0],
                        matrixes[1],
                        self.bias_hidden,
                        self.bias_output) )

    def prediction(self, inputs_array): #Activision Function

        input = np.array(inputs_array).reshape(len(inputs_array), 1)

        # Multiply Matrixes of Input Layer && Hidden Layer
        inputs_mul_weights = np.dot(self.weights_input_hidden, input) #Matrix multuply: (2, 3) * (3, 1) = (2, 1)
        input_weight_add_bias = np.add(inputs_mul_weights, self.bias_hidden) #Matrix sum: (2, 1) + (2, 1) = (2, 1)
        hidden = mapMatrix(input_weight_add_bias, sigmoid) #Sigmod (1/1+e^-n) ==> matrix[i]

        # Multiply Matrixes of Hidden Layer && Output Layer
        hidden_mul_output = np.dot(self.weights_hidden_output, hidden) #Matrix multuply: (2, 3) * (3, 1) = (2, 1)
        hidden_output_add_bias = np.add(hidden_mul_output, self.bias_output)  # Matrix sum: (2, 1) + (2, 1) = (2, 1)
        output = mapMatrix(hidden_output_add_bias, sigmoid) #Sigmod (1/1+e^-n) ==> matrix[i]

        return np.asarray(output).reshape(-1)  # [0] np.array / [1] 1D array



    def train(self, inputs_array, labels_array):
        I_array = np.array(inputs_array).reshape(len(inputs_array), 1)

        # Multiply Matrixes of Input Layer && Hidden Layer
        inputs_mul_weights = np.dot(self.weights_input_hidden, I_array)  # Matrix multuply: (2, 3) * (3, 1) = (2, 1)
        input_weight_add_bias = np.add(inputs_mul_weights, self.bias_hidden)  # Matrix sum: (2, 1) + (2, 1) = (2, 1)
        hidden = mapMatrix(input_weight_add_bias, sigmoid)  # Sigmod (1/1+e^-n) ==> matrix[i]

        # Multiply Matrixes of Hidden Layer && Output Layer
        hidden_mul_output = np.dot(self.weights_hidden_output, hidden)  # Matrix multuply: (2, 3) * (3, 1) = (2, 1)
        input_weight_add_bias = np.add(hidden_mul_output, self.bias_output)  # Matrix sum: (2, 1) + (2, 1) = (2, 1)
        output = mapMatrix(input_weight_add_bias, sigmoid)  # Sigmod (1/1+e^-n) ==> matrix[i]

    # Find the minmum Error:
        # y= mx+b
        # Δm = learning_rate * errorⱼ * xᵢ
        # Δb = learning_rate * errorⱼ

    # Using Matrix version calculations:
        # Derivate'(x) = Sigmoid(x) * (1-Sigmoid(x))
        # ΔWᵢⱼ(HO) = learning_rate * Error(vector) * (Sigmoid(output) * (1-Sigmoid(output)) * (Hidden_Input)ᵀ
        # ΔWᵢⱼ(IH) = learning_rate * Hidden_Error(vector) * (Sigmoid(hidden) * (1-Sigmoid(hidden)) * (Input)ᵀ
        #guess = self.activision_func(inputs)[1]


    #Calculate Error = Target (label) - guess

        #Calculate Output Layer Error
        labels_arrays = np.reshape(labels_array,(1,1))
        output_error = np.subtract(labels_arrays, output)

        # Calculate Gradient Descent
        #Claculate ∇Wᵢⱼ(HO):
        gradient = mapMatrix(output, derivativeSigmoid)
        gradient = np.dot(gradient, output_error) # Calculate Error(vector) * (Derivate'(output))
        gradient = np.dot(gradient, self.learning_rate) # Calculate Error(vector) * (Derivate'(output)) * Learning_Rate


        # Calculate Δ Deltas
        hidden_transpose = np.transpose(hidden)
        weights_ho_delta = np.dot(gradient, hidden_transpose)

        #Adjust the weights by Δ
        self.weights_hidden_output = np.add(self.weights_hidden_output, weights_ho_delta)

        #Adjust the bias by its deltas (The Gradient)
        self.bias_output = np.add(self.bias_output, gradient)

        #Calculate Hidden Layer Error
        transposed_weights_hidden_output = np.transpose(self.weights_hidden_output)
        hidden_error = np.dot(transposed_weights_hidden_output, output_error)

        # Claculate ∇Wᵢⱼ(IH):
        hidden_gradient = mapMatrix(hidden, derivativeSigmoid)
        hidden_gradient = np.multiply(hidden_gradient, hidden_error)  # Calculate Error(vector) * (Derivate'(output))
        hidden_gradient = np.multiply(hidden_gradient, self.learning_rate)  # Calculate Error(vector) * (Derivate'(output)) * Learning_Rate



        # Calculate Δ Hidden Deltas
        inputs_transpose = np.transpose(inputs_array)
        weights_ih_delta = np.multiply(hidden_gradient, inputs_transpose)

        # Adjust the weights by Δ
        self.weights_input_hidden = np.add(self.weights_input_hidden, weights_ih_delta)

        #Adjust the bias by its deltas (The Gradient)
        self.bias_hidden = np.add(self.bias_hidden, hidden_gradient)

    def draw(self,DATA,jumps=1):#MatPlotLib Visualization
        partialData = DATA[::jumps]
        fig = plt.figure(figsize=(10, 5))
        bars = ('Is it XOR? [0,1]= 1', '[1,0]= 1', '[1,1]= 0', '[0,0]= 0', 'BLOCK')
        def animate(i):
            if (i< len(partialData) ):
                plt.cla() #Clean last draw
                plt.bar(bars, partialData[i], width = 0.4)
                plt.xticks(bars)
                plt.title('Adjusting Parameters. Attempt: {0}/{1}'.format(i + 1, len(partialData)))
                plt.xlabel('{0:.2f}| {1:.2f} | {2:.2f} | {3:.2f}'.format(partialData[i][0], partialData[i][1], partialData[i][2], partialData[i][3] ))

            else:
                print('Finished....exit in 5 sec')
                time.sleep(50)
                exit()

        plt.get_current_fig_manager().window.wm_geometry("+1000+10")
        animationFunc = FA(plt.gcf(), animate, interval=1)
        plt.tight_layout()

        plt.show()


setup(2,2,1,10000,10)
