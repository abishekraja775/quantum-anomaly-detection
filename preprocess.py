from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(normal_data, test_data, n_features=2):
    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal_data)
    test_scaled = scaler.transform(test_data)

    pca = PCA(n_components=n_features)
    normal_reduced = pca.fit_transform(normal_scaled)
    test_reduced = pca.transform(test_scaled)

    return normal_reduced, test_reduced
