# MF796 Project 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
from kneed import KneeLocator


def find_clusters(price):
    """
    This function use silhouette method to find out the suitable number of clusters.
    """
    
    rt = price.pct_change().mean()*252
    rt = pd.DataFrame(rt)
    rt.columns = ['returns']
    rt['volatility'] = price.pct_change().std()*np.sqrt(252)
    #Standarlize the data for later use.
    scale = StandardScaler().fit(rt)
    X = pd.DataFrame(scale.fit_transform(rt),columns = rt.columns, index = rt.index)
    
    #Here we use silhouette method to find a suitable K.
    K = range(2,15)
    silhouettes = []

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='random')
        kmeans.fit(X)
        silhouettes.append(silhouette_score(X, kmeans.labels_))
    
    # A brief view of situations for each K.
    fig = plt.figure(figsize= (15,5))
    plt.plot(K, silhouettes, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Silhouette score')
    plt.grid(True)
    plt.show()
    
    kl = KneeLocator(K, silhouettes, curve="convex", direction="decreasing")
    # 'kl.elbow' shows the suggested value of K we can use below.
    
    return kl.elbow, X
 

   
def making_clusters(price):
    """
    This function use K-means to cluseter the stocks and return clustered series.
    """
    c, X = find_clusters(price)
    k_means = KMeans(n_clusters=c)
    k_means.fit(X)
    prediction = k_means.predict(X)
    
    #Here's the plot of our data to see how the cluster distribute.
    centroids = k_means.cluster_centers_
    fig = plt.figure(figsize = (18,10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1], c=k_means.labels_, cmap="rainbow", label = X.index)
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)
    plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=10)
    plt.show()
       
    cluster_result = pd.Series(index=X.index, data=k_means.labels_.flatten())
    cluster_result_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
    cluster_result = cluster_result[cluster_result != -1]
    
    # plt.figure(figsize=(12,8))
    # plt.barh(range(len(cluster_result.value_counts())),cluster_result.value_counts())
    # plt.title('Clusters')
    # plt.xlabel('Stocks per Cluster')
    # plt.ylabel('Cluster Number')
    # plt.show()
    
    return cluster_result



def find_coint_pairs(data, significance=0.05):
    """
    This function input the stock data and required significance.
    It implement a loop (using cointegration test) to find out the pairs and pvalue_matrix.
    """
    n = data.shape[1]    
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(7):
        for j in range(i+1, n):
            S1 = data[keys[i]]            
            S2 = data[keys[j]]
            result = ts.coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs



def sort(pairs,price):
    """
    This function input existing pairs and return a dataframe sorted by their p-value.
    """
    df = pd.DataFrame(index=pairs)
    df['pvalue']=0
    for i in range(len(pairs)):
        s1 = price[pairs[i][0]]
        s2 = price[pairs[i][1]]
        score, pvalue, _ = ts.coint(s1, s2)
        df.iloc[i,0] = pvalue
        
    return df.sort_values(by='pvalue')



def makingpairs(price):
    """
    This function use the all the fuction we mentioned above.
    it make a rough clusters depending on the mean and volativity of each stocks' price and find out the pairs.
    Then it return the pairs results and a dataframe sorted by its p-value.
    """
    cluster_result = making_clusters(price)
    cluster_size_limit = 1000
    counts = cluster_result.value_counts()
    ticker_count = counts[(counts>1) & (counts<=cluster_size_limit)]
    ticker = cluster_result[cluster_result == 1].index
    cluster_dict = {}
    for i, clust in enumerate(ticker_count.index):
        tickers = cluster_result[cluster_result == clust].index
        score_matrix, pvalue_matrix, pairs = find_coint_pairs(price[tickers],significance=0.05)
        cluster_dict[clust] = {}
        cluster_dict[clust]['score_matrix'] = score_matrix
        cluster_dict[clust]['pvalue_matrix'] = pvalue_matrix
        cluster_dict[clust]['pairs'] = pairs
    
    pairs = []   
    for cluster in cluster_dict.keys():
        pairs.extend(cluster_dict[cluster]['pairs'])
    df = sort(pairs,price)
    return pairs, df
    

def output(test_pairs):
    p0 = []
    p1 = []
    for i in range(len(test_pairs)):
        p0 += [test_pairs[i][0]]
        p1 += [test_pairs[i][1]]
            
    out = pd.DataFrame(p0)
    out['1'] = p1
    return out
        
    
if __name__ == '__main__':
    

    price = pd.read_csv('price.csv',index_col='Date')
    price = price.fillna(method = 'ffill')
    price = price.fillna(method = 'bfill')
    
 
    #2015/1/2, 2015/12/31, 2016/1/4, 2016/12/30, 2017/1/3, 2017/12/29
    #2018/1/2, 2018/12/31, 2019/1/2, 2019/12/31, 2020/1/2, 2020/12/31
    #2021/1/4, 2021/12/31
    price_train = price['2015/1/2':'2016/12/30']
    pairs, df = makingpairs(price_train) 
    price_test = price['2017/1/3':'2017/12/29']
    df2 = sort(pairs,price_test)
    test_pair = df2[df2['pvalue']<=0.05].index
    out = output(test_pair)
    out.to_csv('pairs_2017.csv')
    
    # price_train = price['2016/1/4':'2017/12/29']
    # pairs, df = makingpairs(price_train) 
    # price_test = price['2018/1/2':'2018/12/31']
    # df2 = sort(pairs,price_test)
    # test_pair = df2[df2['pvalue']<=0.05].index
    # out = output(test_pair)
    # out.to_csv('pairs_2018.csv')
    
    # price_train = price['2017/1/3':'2018/12/31']
    # pairs, df = makingpairs(price_train) 
    # price_test = price['2019/1/2':'2019/12/31']
    # df2 = sort(pairs,price_test)
    # test_pair = df2[df2['pvalue']<=0.05].index
    # out = output(test_pair)
    # out.to_csv('pairs_2019.csv')
    
    # price_train = price['2018/1/2':'2019/12/31']
    # pairs, df = makingpairs(price_train) 
    # price_test = price['2020/1/2':'2020/12/31']
    # df2 = sort(pairs,price_test)
    # test_pair = df2[df2['pvalue']<=0.05].index
    # out = output(test_pair)
    # out.to_csv('pairs_2020.csv')
    
    # price_train = price['2019/1/2':'2020/12/31']
    # pairs, df = makingpairs(price_train) 
    # price_test = price['2021/1/4':'2021/12/31']
    # df2 = sort(pairs,price_test)
    # test_pair = df2[df2['pvalue']<=0.05].index
    # out = output(test_pair)
    # out.to_csv('pairs_2021.csv')
    
    
    
    
    
    
 
    
 
    