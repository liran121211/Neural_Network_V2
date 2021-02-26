from Perceptrons_Network import *




def setup(inputNodes = 2, hiddenNodes = 2, outputNodes = 1, iterations = 500,jumps = 1):
    NN = NeuralNetwork(inputNodes,hiddenNodes,outputNodes) #Init NeuralNetwork Object
    print(Fore.BLUE + 'Initializing Neural Network ({0}X{1}X{2})'.format(inputNodes, hiddenNodes, outputNodes))
    time.sleep(1)
    print(Fore.BLUE + 'Calculating Matrixes of DATA...')
    DATA = [{ 'inputs': [0, 1] , 'target': [1] },
            { 'inputs': [1, 0] , 'target': [1] },
            { 'inputs': [0, 0] , 'target': [0] },
            { 'inputs': [1, 1] , 'target': [0] }
            ]

    #Matrix compatibility test
    for index in range(len(DATA)):
        if ( len(DATA[index]['inputs']) != inputNodes):
            print (Fore.RED + '[Input Nodes] value must be the same as [{0}] lenght ({1} != {2})'.format(DATA[index]['inputs'], inputNodes, len(DATA[index]['inputs'])))
            exit()

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

    print(Fore.GREEN + 'Done Filling MatPlotLib Array...\n Initiating Graphics...')
    print(Fore.GREEN + 'Final Predictions: {0:.2f}|{1:.2f}|{2:.2f}|{3:.2f}'.format(graphicArray[iterations-1, 0],
                                                               graphicArray[iterations-1, 1],
                                                               graphicArray[iterations-1, 2],
                                                               graphicArray[iterations-1, 3]) )
    NN.draw(graphicArray,jumps)
    return NN, graphicArray


def run():
    setup(2, 10, 6, 10000, 10)


run()