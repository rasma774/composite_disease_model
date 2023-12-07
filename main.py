from keras.layers import Input, Dense, Concatenate
from keras import Model
import numpy as np
import pandas as pd
import os
import argparse

# Authorship and contact information
__author__ = 'Rasmus Magnusson'
__COPYRIGHT__ = 'Copyright (C) 2021 Rasmus Magnusson'
__contact__ = 'rasma774@gmail.com'

# Function to parse command-line arguments
def parse_them_arguments():
    parser = argparse.ArgumentParser(
        description='Select the number of layers and nodes in a DNN'
    )
    parser.add_argument('--Nlayers_funnel', default=[1], type=int, nargs=1, help='Number of layers in the model')
    args = parser.parse_args()

    # Extract the number of layers from the arguments
    Nlayers = args.Nlayers_funnel[0]
    print(Nlayers)
    return Nlayers


def main():
    # Read the data from CSV files
    terms = pd.read_csv('disgenet_pathway.csv', index_col=1)
    with open('data_example.csv', 'r') as f:
        genes = f.readline().strip('\n').split(',')
    assert np.all(terms.iloc[:, 0].isin(genes))

    # Parse command-line arguments to get the number of layers
    n_layers = parse_them_arguments()

    # Build inputs based on unique terms in the data
    fixstr = lambda x: x.replace('(', '').replace(')', '').replace('\'', '').replace(' ', '').replace(',', '')
    inputs = []
    for term in terms.index.unique():
        nGenes_tmp = len(terms.loc[term])
        inp = Input(shape=(nGenes_tmp), name=fixstr(term))
        inputs.append(inp)

    # Build individual funnels (middle layers) between inputs and the concat layer
    middle_layer_funnels = []
    n_nodes = [20 for _ in range(n_layers)]
    n_nodes.append(1)
    for i in range(len(n_nodes)):
        layers_tmp = []
        for j in range(len(inputs)):
            if i == 0:
                last_layer = inputs
            else:
                last_layer = middle_layer_funnels[-1]
            tmp = Dense(n_nodes[i], activation='elu')(last_layer[j])
            layers_tmp.append(tmp)
        middle_layer_funnels.append(layers_tmp)

    concat = Concatenate()(middle_layer_funnels[-1])

    # Add additional dense layers
    n_nodes_dense = [250]
    last_layer = concat
    for n_nodes_tmp in n_nodes_dense:
        last_layer = Dense(n_nodes_tmp, activation='elu')(last_layer)

    # Add the output layer
    output = Dense(len(genes), activation='elu')(last_layer)

    # Create the model
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer='adam',
        loss='MSE',
        metrics=['accuracy']
    )

    # Training loop
    while True:
        data = pd.read_csv('data_example.csv')  # In the paper, all training data were used instead

        # Build the input structure for the model
        input_structure = {}
        for trm_tmp in terms.index.unique():
            genes_tmp = terms.loc[trm_tmp].iloc[:, 0].values
            input_structure[fixstr(trm_tmp)] = data[genes_tmp].values

        output_data = data.values

        # Train the model
        model.fit(
            input_structure,
            output_data,
            epochs=140,
            batch_size=40,
            validation_split=0.1,
        )

        # Save the trained model
        model.save('model_name.h5')


if __name__ == '__main__':
    main()

