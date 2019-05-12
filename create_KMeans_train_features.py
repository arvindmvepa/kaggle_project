from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

def create_KMeans_train_features(X_train_scaled, y_tr, K, maxiter, miniBatchKMeans_option=1, bsize=100):
    """
    This creates new features using K-Means, where each feature contains the distance from a given row to each cluster

    Parameters
    ----------
    X_train_scaled : dataframe
        A dataframe containing the scaled feature vectors
    y_tr: dataframe
        A dataframe containing the scaled outcome vector
    K: int
        The number of clusters
    maxiter: int
       maximum number of iterations for regular K-Means (not miniBatchKMeans)
    miniBatchKMeans_option : int
		Option to use miniBatchKMeans (miniBatchKMeans_option=1) or 
		regular KMeans (miniBatchKMeans_option!=1)
    bsize: int
		batch_size for miniBatchKMeans

    Returns
    -------
    cluster_X_train_scaled_df : Pandas.DataFrame 
        A DataFrame of the X_train_scaled and the distances to each Cluster each row belongs to
		(contains X_train_scaled + K new features)
		
    #centroids_df : Pandas.DataFrame
    #	A DataFrame of the centroid coordinates

    #more effective to use PCA before K-Means because 
	#distance metric doesn't work well for large dimensions?

	#miniBatchKMeans explanation: https://algorithmicthoughts.wordpress.com/2013/07/26/machine-learning-mini-batch-k-means/
	#MiniBatchKMeans is much faster than regular K-Means as it only uses subset of the data
	#It can be effective for large datasets
	"""


	### dataset needs to be scaled before using KMeans ###
	if miniBatchKMeans_option==1:
		kmeans = MiniBatchKMeans(n_clusters=K, batch_size=bsize)
		kmeans = kmeans.fit(X_train_scaled)

	else:
		kmeans = KMeans(n_clusters=K, max_iter=maxiter, algorithm = 'auto') 
		kmeans = kmeans.fit(X_train_scaled)


	cluster_labels=kmeans.labels_
	centroids=kmeans.cluster_centers_

	#create K new features, where each feature contains the squared distance 
	#from each row to each cluster center. X_dist consists of K columns
	X_dist = kmeans.transform(X_train_scaled)**2
	cols=[]

	#column names are Cluster1, Cluster2, ..., ClusterK
	for i in range(1,K+1):
	    coli='Cluster'+str(i)
	    cols.append(coli)
	X_dist_df.columns=cols


	#predicting centroids
	#new_cluster_centers = kmeans.cluster_centers_[kmeans.predict(X_train_scaled)]

	'''
	error = 0
	for i in range(len(X_train_scaled)):
	    #predict the cluster each data row belongs to
	    predicted_clusters = kmeans.predict(X_train_scaled)
	    error+=abs(predicted_clusters -y_scaled)

	print(error)
	'''

	cluster_labels_df=pd.DataFrame(cluster_labels)
	cluster_labels_df.columns = ['Cluster']
	#cluster_X_train_scaled_df = pd.concat([cluster_labels_df, X_train_scaled], axis=1)
	cluster_X_train_scaled_df = pd.concat([X_dist_df, X_train_scaled], axis=1)
	
	centroids_df=pd.DataFrame(centroids)
	centroids_df.columns=X_train_scaled.columns.values


	#save as pickle
    filename = 'KMeans-clusterLabels.pickle'
    outfile = open(filename,'wb')
    pickle.dump(cluster_X_train_scaled_df,outfile)
    outfile.close()

    '''
    filename = 'KMeans-centroids.pickle'
    outfile = open(filename,'wb')
    pickle.dump(centroids_df,outfile)
    outfile.close()
    '''

    #save as CSV
    cluster_X_train_scaled_df.to_csv('KMeans-FeatureVectors.csv', index=False)
    #centroids_df.to_csv('KMeans-centroids.csv')

    
    return cluster_X_train_scaled_df #, centroids_df