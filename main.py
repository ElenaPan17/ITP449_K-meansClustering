""" Elena Pan
    ITP-449
    Assignment 7
    Clustering Wine Quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def main():
    # Step1: read the dataset into a DataFrame
    file = 'wineQualityReds.csv'
    df = pd.read_csv(file)

    # Step2: drop wine from the DataFrame
    df.drop('Wine', axis = 1, inplace = True)
    
    # Step3: extract quality and store it in a seperate DataFrame
    # quality is the target y
    df_new = df[['quality']]

    # Step4: drop quality from the DataFrame
    df.drop('quality', axis = 1, inplace = True)
    
    # Step5: normalize all columns of the DataFrame using StandardScaler
    scaler = StandardScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # Step6
    # create a range of k from 1 to 11
    # iterate on k values and store the inertia_
    ks = range(1, 11)
    inertias = []
    for k in ks:
        model_kmeans = KMeans(n_clusters=k, random_state = 2021)
        model_kmeans.fit(df_norm)
        inertias.append(model_kmeans.inertia_)

    # Step7
    # plot the chart of inertia vs number of clusters k
    plt.plot(ks, inertias, marker = 'o')
    plt.title('Inertia vs Number of Clusters')
    plt.xticks(ks)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.savefig('Clustering.png')
    
    # Step8
    # use Elbow Method to find the optimal K = 7 
    # as the plot has a elbow at k = 7

    # Step9
    # k = 7; random_state = 2021 to instantiate the KMeans model 
    # assign the respective cluster number to each wine
    model = KMeans(n_clusters= 7, random_state= 2021)
    model.fit(df_norm)
    labels = model.predict(df_norm)
    df_norm['Cluster_Number'] = pd.Series(labels)
    
    # Step10
    # add the quality back to the DataFrame
    df_wine = pd.concat([df_norm, df_new], axis = 1)
    # print(df_wine)
    
    # Step11
    # print a crosstab of cluster number vs quality
    print(pd.crosstab(df_wine.Cluster_Number, df_wine.quality))
    
    # Step12 
    # from the crosstable, the clusters cannot represent the quality of the wine 
    # because for each cluster, they mainly gather at quality 5 or 6
    # they do not correspondingly/seperately associate with each of the quality 


if __name__ == '__main__':
    main()
