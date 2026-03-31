import sys
from random import random
import numpy as np
from tqdm import tqdm
import os




def gen_1():
    """ Generates training data which will keep the pad at the same y as the ball

    :return: None
    """
    data = {
        "input": [],
        "keys": []
    }

    print("Generating training data")
    for i in tqdm(range(100000)):
        ball_pos = random()
        pad_pos = random()
        data["input"].append(np.array([ball_pos, pad_pos]))
        if ball_pos < pad_pos:
            data["keys"].append([0])
        else:
            data["keys"].append([1])
    #dump training set
    with open('training_set.pkl', 'wb+') as file:
        pickle.dump(data, file)


print(os.listdir("./collected_data/"))

def gen_2():
    """ Will generate training data based on collected data. Will make the ai predict where the ball will land

    :return: None
    """
    for i in range(len(os.listdir("./collected_data/"))):
        training_data ={
            "input": [],
            "keys": []
        }

        with open(f"./collected_data/set_{i}.npy", "rb") as file:
            results = np.load(file)
            for j in range(len(results)):
                current_result = results[j]
                collection = np.load(file)
                for input_values in collection:
                    training_data["input"].append(input_values)
                    random_pad_pos = input_values[4]
                    if current_result < random_pad_pos:
                        training_data["keys"].append([0])
                    else:
                        training_data["keys"].append([1])

        training_data["input"] = np.array(training_data["input"], dtype=np.float32)
        training_data["keys"] = np.array(training_data["keys"], dtype=np.float32)
        print(f"Set {i}: {training_data['input'].shape}")
        with open(f"./gen_2_training_sets/set_{i}.npy", "wb+") as file:
            np.save(file, training_data["input"])
            np.save(file, training_data["keys"])



gen_2()

