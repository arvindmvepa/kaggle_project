def adapt_data(X, Y, pca_n_components=None, pca_exp_var=None, kmeans_n_clusters=None,
	miniBatchKMeans_option=1, bsize=100):
	new_vectors = pd.DataFrame()
    if pca_n_components or pca_exp_var:
    	# apply PCA decision logic
        pca_vectors = create_PCA_train_features(X, Y, pca_n_components, pca_exp_var):

        if pca_exp_var:
           # exp var logic
           #new_vectors += pca_vectors
           new_vectors = pd.concat([pca_vectors, new_vectors], axis=1)
        else:
           # n_components logic
           #new_vectors += pca_vectors
           new_vectors = pd.concat([pca_vectors, new_vectors], axis=1)

    if kmeans_n_clusters:
        kmeans_vectors = create_KMeans_train_features(X, Y, kmeans_n_clusters, maxiter, 
            miniBatchKMeans_option, bsize)

        new_vectors = pd.concat([kmeans_vectors, new_vectors], axis=1)

    X = pd.concat([new_vectors, X], axis=1)

    return X, Y