from sklearn.decomposition import PCA
import pickle

def create_PCA_train_features(X_train_scaled, y_tr, N, pca_exp_var):
    """
    This converts the feature vectors and outcome vector into the first N Principal Components

    Parameters
    ----------
    X_train_scaled : dataframe
        A dataframe containing the scaled feature vectors
    y_tr: dataframe
        A dataframe containing the scaled outcome vector
    N: int
        The number of Principal Components
    pca_exp_var: float
        Minimum desired cumuluative variance


    Returns
    -------
    X_rPCA_df : Pandas.DataFrame
        A DataFrame of the Principal Components

    #X_rPCA_scaled, y_rPCA, y_rPCA_scaled : Pandas.DataFrame
    #    A DataFrame of the Principal Components
    #scaler : StandardScaler()

    """

    #print("X_train_scaled shape: ", X_train_scaled.shape)

    #need to use numpy array for PCA
    Ymat = y_tr.as_matrix()
    Xmat = X_train_scaled.as_matrix()

    if pca_exp_var:
        totalvar=0
        Npca=0
        while totalvar<pca_exp_var:
            pca = PCA(n_components=Npca)
            X_rPCA = pca.fit(Xmat).transform(Xmat)
            y_rPCA = pca.fit(Ymat).transform(Ymat)

            #print('PCA explained variance ratio (first %d components): %s' % (N, pca.explained_variance_ratio_))
            #variance = pca.explained_variance_ratio_ #calculate variance ratios
            totalvar=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
            print('cumulative variance PCA: %f' % totalvar) #cumulative sum of variance explained with [n] features
            Npca+=1
    else: #use n_components=N specified by user
        pca = PCA(n_components=N)
        X_rPCA = pca.fit(Xmat).transform(Xmat)
        y_rPCA = pca.fit(Ymat).transform(Ymat)

        print('PCA explained variance ratio (first %d components): %s' % (N, pca.explained_variance_ratio_))
        var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
        print('cumulative variance PCA: %f' % var) #cumulative sum of variance explained with [n] features


    X_rPCA_df=pd.DataFrame(X_rPCA)
    X_rPCA_df.to_csv('PCA-FeatureVectors.csv', index=False)

    #save as pickle
    filename = 'PCA-FeatureVectors.pickle'
    outfile = open(filename,'wb')
    pickle.dump(X_rPCA_df,outfile)
    outfile.close()

    '''
    # scale feature vector
    scaler = StandardScaler()
    scaler.fit(X_rPCA)
    X_rPCA_scaled = pd.DataFrame(scaler.transform(X_rPCA), columns=X_rPCA.columns)
    y_rPCA_scaled = pd.DataFrame(scaler.transform(y_rPCA), columns=y_rPCA.columns)
    
    #save as CSV
    X_rPCA_df=pd.DataFrame(X_rPCA)
    X_rPCA_df.to_csv('PCA-FeatureVectors.csv', index=False)
    X_rPCA_scaled_df=pd.DataFrame(X_rPCA_scaled)
    X_rPCA_scaled_df.to_csv('PCA-FeatureVectors_scaled.csv', index=False)
    y_rPCA_df=pd.DataFrame(y_rPCA)
    y_rPCA_df.to_csv('PCA-yVectors.csv', index=False)
    y_rPCA_scaled_df=pd.DataFrame(y_rPCA_scaled)
    y_rPCA_scaled_df.to_csv('PCA-yVectors_scaled.csv', index=False)

    #save as pickle
    filename = 'PCA-FeatureVectors.pickle'
    outfile = open(filename,'wb')
    pickle.dump(X_rPCA_df,outfile)
    outfile.close()

    filename = 'PCA-FeatureVectors_scaled.pickle'
    outfile = open(filename,'wb')
    pickle.dump(X_rPCA_scaled_df,outfile)
    outfile.close()

    filename = 'PCA-yVectors.pickle'
    outfile = open(filename,'wb')
    pickle.dump(y_rPCA_df,outfile)
    outfile.close()

    filename = 'PCA-yVectors_scaled.pickle'
    outfile = open(filename,'wb')
    pickle.dump(y_rPCA_scaled,outfile)
    outfile.close()
    '''
    
    return X_rPCA_df #, X_rPCA_scaled, y_rPCA, y_rPCA_scaled, scaler

