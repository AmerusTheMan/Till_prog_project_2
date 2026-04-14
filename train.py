import os
import cupy as np
from network import Network
import pickle




def prepare_data(path):
    """prepare data that can be used to train the ai
    
    :param path: path to training set
    :return: A0(input data), Y(expected output), m(sample size)
    """
    with open(path, "rb") as file:
        X = np.load(file)
        y = np.load(file)

    m = len(y) #training samples


    # we need to reshape y to a n^[L] x m matrix
    Y = y.reshape(ai.nodes[ai.L], m)

    A0 = X.T #transpose the input data

    return A0, Y, m


ai = Network(
    layers = [5, 20, 20, 20, 1],
    model_path="model_temp.pkl"

)



for i in range(100):
    for set_name in os.listdir("./gen_2_training_sets"):
        print(f"training on {set_name}")
        input_data, expected_oputput, sample_size = prepare_data(os.path.join("./gen_2_training_sets", set_name))
        
        ai.train(
            input_data=input_data,
            keys=expected_oputput,
            m=sample_size,
            epochs=10000,
            alpha=0.1
        )

        y_hat = ai.calc_output(A0)
        print(Y)
        print(y_hat)

        calculated_cost = ai.cost(y_hat, Y)
        print(f"cost after: {calculated_cost}")

        

    
    

