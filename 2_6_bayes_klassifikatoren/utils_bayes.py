import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


newsgroups_categories = {
    "Religion2": 'alt.atheism',
    "Computer": 'comp.graphics',
    "Computer2": 'comp.os.ms-windows.misc',
    "Computer3": 'comp.sys.ibm.pc.hardware',
    "Computer4": 'comp.sys.mac.hardware',
    "Computer5": 'comp.windows.x',
    "Verkauf": 'misc.forsale',
    "Autos": 'rec.autos',
    "Motorräder": 'rec.motorcycles',
    "Baseball": 'rec.sport.baseball',
    "Hockey": 'rec.sport.hockey',
    "Kryptographie": 'sci.crypt',
    "Elektronik": 'sci.electronics',
    "Medizin": 'sci.med',
    "Space": 'sci.space',
    "Religion": 'soc.religion.christian',
    "Waffen (US)": 'talk.politics.guns',
    "Mittlerer Westen": 'talk.politics.mideast',
    "Politik": 'talk.politics.misc',
    "Religion": 'talk.religion.misc'
}


def plot_gaussian_nb(X, y):

    fig, ax = plt.subplots()
    fig.set_size_inches((10, 8))

    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
    ax.set_title('Naive Bayes Model', size=14)

    xlim = (-8, 8)
    ylim = (-15, 5)

    xg = np.linspace(xlim[0], xlim[1], 60)
    yg = np.linspace(ylim[0], ylim[1], 40)
    xx, yy = np.meshgrid(xg, yg)
    Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

    for label, color in enumerate(['red', 'blue']):
        mask = (y == label)
        mu, std = X[mask].mean(0), X[mask].std(0)
        P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
        Pm = np.ma.masked_array(P, P < 0.03)
        ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5,
                      cmap=color.title() + 's')
        ax.contour(xx, yy, P.reshape(xx.shape),
                   levels=[0.01, 0.1, 0.5, 0.9],
                   colors=color, alpha=0.2)

    ax.set(xlim=xlim, ylim=ylim)
    
    return fig, ax


def get_multinomial_nb_model(themen=None):
    if themen is None:
        themen = ["Space", "Hockey", "Baseball", "Computer"]
        
    categories = [newsgroups_categories[thema] for thema in themen]

    # train = fetch_20newsgroups(subset="train", categories=categories)
    # test = fetch_20newsgroups(subset="test", categories=categories)
    data = fetch_20newsgroups(categories=categories)
    
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(data.data, data.target)
    
    def predict_category(s, data=data, model=model):
        pred = model.predict([s])
        cat = data.target_names[pred[0]]
        
        for thema, category in newsgroups_categories.items():
            if category == cat:
                return thema
    
    return model, predict_category




