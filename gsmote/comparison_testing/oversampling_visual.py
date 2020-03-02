import datetime
from gsmote import GSMOTE as gs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import gsmote.comparison_testing.preprocessing as pp

# Partition the dataset
from sklearn.model_selection import train_test_split

date_file = "../../data/KDD.csv"
X,y = pp.pre_process(date_file)

X, X_t, y, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Instantiate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(X)


X_resampled, y_resampled = gs.OverSample(X, y)
X_res_vis = pca.transform(X_resampled)

# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2)

c0 = ax1.scatter(X_vis[y == '0', 0], X_vis[y == '0', 1], label="Class #0",
                 alpha=0.5,marker='.')
c1 = ax1.scatter(X_vis[y == '1', 0], X_vis[y == '1', 1], label="Class #1",
                 alpha=0.5,marker='.')
ax1.set_title('Original set')

ax2.scatter(X_res_vis[y_resampled == '0', 0], X_res_vis[y_resampled == '0', 1],
            label="Class #0", alpha=0.5, marker='.')
ax2.scatter(X_res_vis[y_resampled == '1', 0], X_res_vis[y_resampled == '1', 1],
            label="Class #1", alpha=0.5,marker='.')
ax2.set_title('GSMOTE OverSampling')

# make nice plotting
for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([-6, 8])
    ax.set_ylim([-6, 6])

f.legend((c0, c1), ('Class #0', 'Class #1'), loc='lower center',
         ncol=2, labelspacing=0.)
plt.tight_layout(pad=3)
# plt.show()
plt.savefig("../../output/oversampled_" + datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S") + ".png", dpi=2000)