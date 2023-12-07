import pandas as pd

__author__ = 'Rasmus Magnusson'
__COPYRIGHT__ = 'Copyright (C) 2021 Rasmus Magnusson'
__contact__ = 'rasma774@gmail.com'


THRESH = 10

def main():
    df = pd.read_csv('curated_gene_disease_associations.tsv', sep='\t')
    df = df[['geneSymbol', 'diseaseName']]
    df = df[df.iloc[:, 0].isin(get_filter())]

    df_occurances = df.iloc[:, 1].value_counts()
    keep = df_occurances.index[df_occurances >= THRESH]

    df[df.iloc[:, 1].isin(keep)].to_csv('disgenet_pathway.csv', index=None)
main()
