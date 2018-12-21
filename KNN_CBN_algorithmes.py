import logging
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn.metrics.pairwise import euclidean_distances
from functools import reduce
from sklearn.neighbors import KNeighborsClassifier

def error_classification(arg1, arg2):
    
    assert type(arg1) == type(arg2)

    size = len(arg1)
    if size != len(arg2):
        logging.error("Liste de taille differente")
        return 0

    SUM = 0
    for i in range(size):
        if arg1[i] != arg2[i]:
            SUM += 1

    return SUM*100 / float(size)

def PPV(x, y, k):
    
    data_size = len(x)

    # ppv_target contient les nouvelles etiquettes
    ppv_target = np.zeros(data_size, dtype=np.int)

    # Cross validation
    for i in range(data_size):
        valide = x[i]
        train = np.delete(x, i, 0)
        train_target = np.delete(y, i, 0)
        assert data_size-1 == len(train)
        
        dist = list(euclidean_distances(train, valide.reshape(1, -1)))
        
        #Sélectionnez la classe des données les plus proches
        if k == 1:
            min_dist_idx = np.argmin(dist)
            ppv_target[i] = train_target[min_dist_idx]
        else:
            # Tri croissant 
            #-> recupération du premier k 
            #-> recupération des index
            
            dist_trie = np.sort(dist, kind='heapsort', axis=0)
            kfirst_dist = dist_trie[0:k]
            kfirst_dist_index = [dist.index(e) for e in kfirst_dist]

            # Compter le nombre d'éléments par classe
            class_dist = Counter([train_target[e] for e in kfirst_dist_index])

            # Recupération de la classe majoritaire
            MAX = -1
            idex_max = -1
            for key, value in class_dist.items():
                if value > MAX:
                    MAX = value
                    idex_max = key

            ppv_target[i] = idex_max
        
    return {'target': ppv_target, 'error': error_classification(ppv_target, y)}

def CBN(x, y):

    nb_class = np.unique(y).size #la taille des etiquettes généré par iris
    data_size = len(x)
    #nb_features = x.shape[1]
    proba_class = [e / float(len(x)) for e in Counter(y).values()]
    
    print(proba_class)

    # cbn_target represente les nouvelles etiquettes
    cbn_target = np.zeros(data_size, dtype=np.int)

    # Cross validation
    # Choisi une observation parmis les donnees
    # Genere la matrice d'apprentissge en excluant l'observation precedente
    for i in range(data_size):
        valide = x[i]
        train = np.delete(x, i, 0)
        train_target = np.delete(y, i, 0)
        assert data_size-1 == len(train)

        # Regroupe les donnees par classe et calcule leurs barycentres
        data_par_class = {}
        mean_par_class = {}
        for j in range(nb_class):
            data_par_class[j] = [e for k, e in enumerate(train) if train_target[k] == j]
            mean_par_class[j] = np.mean(a=data_par_class[j], axis=0)

        # Calcule des distances
        # dist represente la distance entre la donnee x et le barycentre pour chaque classe
        # dist_total est la somme des distances par classe
        dist = [abs(valide-barycentre) for barycentre in mean_par_class.values()]
        dist_total = np.sum(dist, axis=0)

        # Calcule de la probabilite PROD(P(xi/wk)P(wk))
        # xi/wk = une donnée x avec la valeur xi pour la variable i de la classe wk
        tmp = (1-(dist/dist_total))*(1./3)
        tmp = [reduce(lambda x, y: x*y, value) for value in tmp]
        
        #print("TMP : %s", tmp)
        #print("TEST : %s", np.argmax(tmp))

        cbn_target[i] = np.argmax(tmp)

    return {'target': cbn_target, 'error': error_classification(cbn_target, y)}

def predic_cross_valid(algo, x, y):
    
    data_size = len(x)

    predicted_target = np.zeros(data_size, dtype=np.int)

    for i in range(data_size):
        valide = x[i]
        train = np.delete(x, i, 0)
        train_target = np.delete(y, i, 0)

        algo.fit(train, train_target)
        predicted_target[i] = algo.predict(valide.reshape(1, -1))

    return {'target': predicted_target, 'error': error_classification(predicted_target, y)}


def main():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    k = 1
    
    #================================Plus Proche Voisin======================================
    #========================================================================================
   
    print('\n================================Plus Proche Voisin======================================')
    print('========================================================================================\n')
    
    ppv = PPV(x=X, y=Y,k=k)
    
    print('==> calcule d\'étiquette prédite pour chaque donnée avec l\'erreur de prédiction:\n')
    print("\tétiquette prédite: ",ppv['target'])
    print("\t\npourcentage d\'étiquettes mal prédites.: ",ppv['error'])
    
    print('\n************************************************************************************')
    print('************************************************************************************\n')
    
    print('==> la fonction des K Plus Proches Voisins de sklearn (avec K = 1)\n')
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X, Y) 
    print("étiquette prédite: ",neigh.predict(X))
    
    #Visualisation 
    print('\n==> Visualisation\n')
    plt.subplot(121)
    plt.title('les etiquettes générées')
    plot_style = [{'c': 'green', 'marker': 'o'},
                        {'c': 'red',   'marker': 'o'},
                        {'c': 'blue',  'marker': 'o'},
                        {'c': 'cyan',  'marker': 'o'}]
    for i in range(len(Y)):
        plt.scatter(X[:,0],X[:,1],Y, **plot_style[Y[i]])
    
    print('\n')
    plt.subplot(122)
    plt.title('les etiquettes prédites')
    for i in range(len(ppv['target'])):
        plt.scatter(X[:,0],X[:,1], ppv['target'], **plot_style[ppv['target'][i]])

   #===============================Classifieur Bayesien Naif=================================
   #=========================================================================================
    print('\n===============================Classifieur Bayesien Naif:=================================')
    print('=========================================================================================\n')
    
    cbn_target = CBN(x=X, y=Y)
    print("CBN : ", cbn_target['target'])
    print("\nCBN erreur : ", cbn_target['error'])
    
    print('\n==> Testez la fonction du Classifieur Bayesien Naïf inclut dans sklearn')
    nbg = naive_bayes.GaussianNB()
    y_pred = nbg.fit(X, Y).predict(X)
    print("\nNaive Bayes GaussianNB avec sklearn: ",y_pred)
    print("Number of mislabeled points out of a total %d points : %d" % (X.shape[0],(Y != y_pred).sum()))
    print("\nNBG avec sklearn erreur : ", error_classification(y_pred, Y))
    
    nbg_target = predic_cross_valid(algo=nbg, x=X, y=Y)
    print('\n\n')
    print("\nNaive Bayes GaussianNB : ", nbg_target['target'])
    print("\nNBG erreur : ", nbg_target['error'])

if __name__ == '__main__':
    main()
