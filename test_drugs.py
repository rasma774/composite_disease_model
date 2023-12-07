from keras.models import load_model, Model
from keras.layers import Input
import pandas as pd
import numpy as np
from numpy.matlib import repmat


__author__ = 'Rasmus Magnusson'
__COPYRIGHT__ = 'Copyright (C) 2021 Rasmus Magnusson'
__contact__ = 'rasma774@gmail.com'


fixstr = lambda x: x.replace('(', '').replace(')','').replace('\'', '').replace(' ', '').replace(',', '')


def build_input(data, annots):
    input_structure = {}
    for trm_tmp in annots.index.unique():
        genes_tmp = annots.loc[trm_tmp].iloc[:,0].values
        tmp = data.transpose().reindex(genes_tmp).values
        tmp[np.isnan(tmp)] = 0
        input_structure[fixstr(trm_tmp)] = tmp.T
    return input_structure

def get_models():
    model = load_model('model_funnel_layers1.h5')
    names = np.array([layer.name for layer in model.layers[:len(model.input_shape)]])
    dem_inputs = [layer.input for layer in model.layers[:len(model.input_shape)]]
    dem_inputs = [layer.input for layer in model.layers[:len(model.input_shape)]]

    m_enc = Model(inputs=dem_inputs, outputs=model.layers[-3].output)

    inp_dec = Input(shape=(model.layers[-2].input_shape[1]))
    dec = inp_dec
    for layer in model.layers[-2:]:
        dec = layer(dec)
    m_dec = Model(inputs=inp_dec, outputs=dec)
    return m_enc, m_dec, names

def get_latent(m_enc, input_structure):
    latent_encoding = m_enc.predict(input_structure)
    my = latent_encoding.mean(0)
    sigma = latent_encoding.std(0)
    return my, sigma,

def perturb_latent_space(enc_mean, enc_sigma, m_dec):
    means = repmat(enc_mean, len(enc_mean), 1)
    means[np.diag_indices(enc_mean.shape[0])] = means.diagonal() + 5*(enc_sigma)

    prediction_background = m_dec.predict(enc_mean.reshape(1, -1))
    perturb_responses = m_dec.predict(means)
    return np.abs(perturb_responses - prediction_background)

def read_data():
    data = pd.read_csv(
        '1902_merged_gene_counts_rn2.txt',
        sep='\t',
        index_col=0,
        )

    key = pd.read_csv(
        'map_IDs.csv',
        index_col=0,
        )

    data = pd.merge(key, data, left_on='ENSEMBL', right_index=True)
    data = data.set_index('SYMBOL')

    data = data.groupby('SYMBOL').sum()
    data = np.log(data + 1)
    data_ET = data.iloc[:, ['ET_' in x for x in data.columns]]
    data_ctrl = data.iloc[:, ['ET_' not in x for x in data.columns]]
    return data_ET, data_ctrl


def disease_drug_associations():
    fname = '41467_2016_BFncomms10331_MOESM423_ESM.xls'
    sheet = pd.read_excel(fname, header=1)
    sheet = sheet.iloc[:, [0, 2]].set_index('Disease').dropna()
    return sheet

def get_drug_targets():
    drug_targets = pd.read_csv(
        'pharmacologically_active.csv',
        )

    drug_targets = drug_targets[drug_targets.Species == 'Humans']

    drug_targets = drug_targets[['Gene Name', 'Drug IDs']]
    drug_targets.iloc[:, -1] = drug_targets.iloc[:, -1].str.split('; ')
    drug_targets = drug_targets.explode('Drug IDs')
    drug_targets = drug_targets.set_index('Drug IDs')
    return drug_targets

# First, we read in a validation set to the data. Build the input structure
# needed for the disgenet-model
data = pd.read_csv('val_data.csv')
annots = pd.read_csv('disgenet_pathway.csv', index_col=1)
input_structure = build_input(data, annots)

# We also read the data of the endothelin1 and ctrl measurements
data_ET, data_ctrl = read_data()

# Now, we read in the model, splitted up between encoders and decoders
m_enc, m_dec, names = get_models()

# We analyse the latent space to find the mean and standard deviation of the
# validation data
enc_mean, enc_sigma = get_latent(m_enc, input_structure)


input_structure_ET = build_input(data_ET.transpose(), annots)
input_structure_ctrl = build_input(data_ctrl.transpose(), annots)

# Now we check the mean values of the latent space in the endothelin1 and control
mET, _ = get_latent(m_enc, input_structure_ET)
mctrl, _ = get_latent(m_enc, input_structure_ctrl)

# Since we have estimated the sigmas since before, we can estimate Z-scores
Zscores = np.abs((mET - mctrl)/ enc_sigma)
Zscores[np.isnan(Zscores)] = 0 #Dont worry, some genes have std = 0 because of nodes that more or less died out

# Also test to put condition that enc_sigma > 0.1
Zscores_corr = Zscores.copy()
Zscores_corr[enc_sigma < .1] = 0
diseases = np.array(m_enc.input_names)[np.argsort(Zscores_corr)]
pd.Series(dict(zip(m_enc.input_names, Zscores_corr))).sort_values()[::-1].to_csv('differentially_activated_nodes_in_endothelin1_stimulated_cells.csv')

# We do the perturbations to the latent space
lat_pert = perturb_latent_space(enc_mean, enc_sigma, m_dec)

lat_pert_tmp = lat_pert[np.argsort(Zscores_corr), :]
lat_pert_tmp = pd.DataFrame(lat_pert_tmp)
lat_pert_tmp.columns = data.columns

# Now we also read in the disease_genes!
drug_targets = get_drug_targets()

tmp = lat_pert_tmp.iloc[-1, :]
fold = {}
for d_id in drug_targets.index.unique():
    target_genes_tmp = drug_targets.loc[d_id].values.T[0]
    if not type(target_genes_tmp) == str:
        fold[d_id] = tmp.reindex(target_genes_tmp).mean()/tmp.mean()
fold = pd.Series(fold)
fold = fold.sort_values()

drugIDs = pd.read_csv('drugbank vocabulary.csv')
drugIDs = drugIDs.iloc[:, [0, 2]]
res = pd.merge(pd.DataFrame(fold), drugIDs, how='left', left_index=True, right_on='DrugBank ID')

coltmp = np.array(res.columns)
coltmp[0] = 'score'
res.columns = coltmp
res = res.iloc[::-1, :]
res.to_csv('disgenet_predicted_drugs_hypertrophy.csv', index=False)

