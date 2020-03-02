"""Class to visualize the data"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

def visualize_data(X,y,title):
    feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None

    # For reproducibility of the results
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    # Principal component analysis for multi-dimensional data
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]

    plt.figure(figsize=(16, 10))
    plt.title(title)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df.loc[rndperm, :],
        legend="full",
        alpha=0.3
    )
    plt.xlim(-6,6)
    plt.ylim(-6,6)

    #plt.show()
    #plt.savefig("../../output/compare_" + datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S") + ".png", dpi=1200)

