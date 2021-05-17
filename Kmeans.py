import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering,KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import io
import sys
from sklearn.metrics import silhouette_score

digits_tra = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)
digits_tes = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes", header=None)

digits_tra = digits_tra.T
digits_tes = digits_tes.T
digits_train = []
for i in range(3823):
    digits_train.append(digits_tra[i])
    
digits_test = []
for i in range(1797):
    digits_test.append(digits_tes[i])

## K Moyennes (implémentation de l'algorithme)

# Dans cette partie, l'algortihme des K Moyennes a été réimplémanté, en conséquence la plupart des fonctions sont lentes.

# impl_Kmeans(n_clusters) lance l'algorithme KMeans pour n_clusters groupes et renvoit différentes informations du clustering obtenu.

# si result = impl_Kmeans(n_clusters),
# silhouette_i(result[2]) renvoit la silhouette du clustering de result
# hist_clusters_i(result[2]) affiche l'histogramme des classes créées par le clustering
# confusion_matrix_i(result[2]) affiche les labels des classes et renvoit la matrice de confusion du clustering (pour les données d'apprentissage)
# confusion_matrix_test_i(res[2],res[3]) attribue un label à chaque donnée de la base de test, affiche les labels des clusters et renvoit la matrice de confusion
# hist_clusters_test_i(result[2],result[3])  attribue un label à chaque donnée de la base de test et affiche l'histogramme des classes créées par le clustering

def closest_center(point,centers) :
    min = distance.euclidean(point,centers[0])
    index_min = 0
    for i in range(1,len(centers)):
        if distance.euclidean(point,centers[i]) < min :
            min = distance.euclidean(point,centers[i])
            index_min = i
    return index_min

def erreur_quadratique_i(clusters,centers) :
    error = 0
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            error = error + distance.euclidean(clusters[i][j],centers[i])
    return error

def impl_Kmeans(k):
    couts = []
    erreur = sys.maxsize
    tries = []
    for i in range(10):
        centers = [digits_train[x][:64] for x in np.random.randint(len(digits_train),size = k)]
        erreur = sys.maxsize
        for j in range(100):
            clusters = [[] for i in range(k)]
            clusters_index = [[] for i in range(k)]
            for m in range(len(digits_train)) :
                center = closest_center(digits_train[m][:64],centers)
                clusters[center].append(digits_train[m][:64])
                clusters_index[center].append(m)
            for i in range(k):
                centers[i] = np.mean(clusters[i],axis = 0)

            couts.append((erreur - erreur_quadratique_i(clusters,centers))/ erreur)
            if ((erreur - erreur_quadratique_i(clusters,centers))/ erreur < 0.0000001 and j > 0):
                print (erreur_quadratique_i(clusters,centers))
                break
            erreur = erreur_quadratique_i(clusters,centers)
            print(erreur)
        tries.append([clusters,erreur,clusters_index,centers])
    min = tries[0][1]
    result = tries[0]
    for i in range(1,10):
        if (tries[i][1] < min):
            min = tries[i][1]
            result = tries[i]
    print("erreur min : " ,min)
    plt.plot(couts)
    result.append(couts)
    return result
    
def get_clusters_classes(clusters_index):
    clusters_classes = []
    for i in range(len(clusters_index)):
        cluster_classes = []
        for j in range(len(clusters_index[i])):
            cluster_classes.append(digits_train[clusters_index[i][j]][64])
        clusters_classes.append(cluster_classes)
    return clusters_classes

def get_ai_bi(point,cluster,clusters_index):
    ai = 0
    l = []
    for i in range(len(clusters_index)):
        bi = 0
        if i == cluster :
            for j in range(len(clusters_index[i])):
                ai += distance.euclidean(digits_train[point],digits_train[clusters_index[i][j]])
            ai = ai/len(clusters_index[i])
        else :
            for j in range(len(clusters_index[i])):
                bi += distance.euclidean(digits_train[point],digits_train[clusters_index[i][j]])
            bi = bi/len(clusters_index[i])
            l.append(bi)
    bi = min(l)
    return (ai,bi)

def silhouette_i(clusters_index):
    l=[]
    for i in range(len(clusters_index)):
        for j in range(len(clusters_index[i])):
            (ai,bi) = get_ai_bi(clusters_index[i][j],i,clusters_index)
            l.append((bi - ai)/ max(ai,bi))
    print(ai,bi)
    return np.mean(l)

def order_classes(clusters_index):
    clusters_classes = get_clusters_classes(clusters_index)
    clusters_label=[]
    for i in clusters_classes:
        clusters_label.append( [i,np.argmax(np.bincount(i))])
    clusters_label.sort(key = lambda t : t[1])
    return clusters_label

def hist_clusters_i(clusters_index):
    clusters_classes = get_clusters_classes(clusters_index)
        
    fig = plt.figure()
    clusters_classes_ordered = [i[0] for i in order_classes(clusters_index)]
    clusters_labels = [i[1] for i in order_classes(clusters_index)]

    for i in range(len(clusters_classes_ordered)):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.hist(clusters_classes_ordered[i],rwidth = 0.1,bins = np.arange(11)-0.5)
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
        ax.text(0,7,str(clusters_labels[i]))
        
    plt.show()

def confusion_matrix_i(clusters_index):
    clusters_classes = get_clusters_classes(clusters_index)
    clusters_label=order_classes(clusters_index)
    for i in clusters_label:
        print(i[1])
    result = np.zeros((10,len(clusters_index)))
    for i in range(len(clusters_label)) :
        for j in clusters_label[i][0]:
            result[j][clusters_label[i][1]]+= 1
    return result
    
def assign_cluster(clusters_index,centers):
    clusters_classes = get_clusters_classes(clusters_index)
    clusters_label=[]
    for i in clusters_classes:
        clusters_label.append( np.argmax(np.bincount(i)))
    print("classes:")
    for i in clusters_label:
        print(i)
    true_classes = []
    assigned_classes = []
    for m in range(len(digits_test)) :
        true_classes.append(digits_test[m][64])
        center = closest_center(digits_test[m][:64],centers)
        assigned_classes.append(clusters_label[center])
    return(true_classes,assigned_classes)

def confusion_matrix_test_i(clusters_index,centers):
    true_classes, assigned_classes = assign_cluster(clusters_index,centers)
    result = np.zeros((10,10))
    for i in range(len(true_classes)):
        result[true_classes[i]][assigned_classes[i]] += 1
    return result
    
def hist_clusters_test_i(clusters_index,centers):
    true_classes, assigned_classes = assign_cluster(clusters_index,centers)
    classes = [[] for i in range(10)]
    for i in range(len(true_classes)):
        classes[true_classes[i]].append(assigned_classes[i])
    
    fig = plt.figure()
    for i in range(len(classes)):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.hist(classes[i],rwidth = 0.1,bins = np.arange(11)-0.5)
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
        ax.text(1,1,"classe réelle : " + str(i),horizontalalignment='right', verticalalignment='top',transform=ax.transAxes)
    plt.show()

## K Moyennes (avec sklearn)

# Dans cette partie, la fonction KMeans de sklearn a été utilisé.
# Les fonctions prennent en argument le nombre de clusters désiré ainsi que inits, le nombre de fois que doit être lancé l'algorithme des K Moyennes.

# erreur_quadratique affiche l'erreur quadratique au fil des itérations de l'algorithme et renvoit l'erreur quadratique finale de la meilleure itération de l'algorithme.
# hist_cluster affiche l'histogramme des classes créées par le clustering
# silhouette renvoit la silhouette du clustering
# confusion_matrix_test affiche les labels des classes et renvoit la matrice de confusion de la base de donnée de test du clustering

def train_kmeans(X,K,inits = 10):
    kmeans = KMeans(n_clusters=K, verbose=2,n_init = inits) 
    kmeans.fit(X)
    return kmeans

def redirect_wrapper(f, inp,K,inits):
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    returned = f(inp,K,inits = inits)               
    printed = new_stdout.getvalue()

    sys.stdout = old_stdout
    return returned, printed


def erreur_quadratique(K,inits = 10):
    returned, printed = redirect_wrapper(train_kmeans, digits_train,K,inits)
    inertia = [float(i[i.find('inertia')+len('inertia')+1:]) if i.find('inertia') != -1 else None for i in printed.split('\n')[1:-2]]
    plt.plot(inertia)
    plt.show()
    return returned.inertia_

def hist_clusters(K,inits = 10):
    kmeans = KMeans(n_clusters = K,n_init = inits)
    kmeans.fit(digits_train)
    cluster_classes = [[] for i in range(K)]
    for i in range(len(kmeans.labels_)):
        cluster_classes[kmeans.labels_[i]].append(digits_train[i][64])
    clusters_classes_label = []
    for i in cluster_classes:
        clusters_classes_label.append([i,np.argmax(np.bincount(i))])
    clusters_classes_label.sort(key = lambda t : t[1])
    fig = plt.figure()
    for i in range(len(clusters_classes_label)):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.hist(clusters_classes_label[i][0],rwidth = 0.1,bins = np.arange(11)-0.5)
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
        ax.text(0,7,str(clusters_classes_label[i][1]))
        
    plt.show()

def silhouette(K,inits = 10):
    kmeans = KMeans(n_clusters = K,n_init = inits)
    kmeans.fit(digits_train)
    return silhouette_score(digits_train,kmeans.labels_)

def confusion_matrix_test(K,inits = 10):
    kmeans = KMeans(n_clusters = K,n_init = inits)
    kmeans.fit(digits_train)
    predictions = kmeans.predict(digits_test)
    cluster_classes = [[] for i in range(K)]
    for i in range(len(predictions)):
        cluster_classes[predictions[i]].append(digits_test[i][64])
    clusters_classes_label = []
    for i in cluster_classes:
        clusters_classes_label.append([i,np.argmax(np.bincount(i))])
    result = np.zeros((10,10))
    for i in range(len(predictions)):
        result[digits_test[i][64]][clusters_classes_label[predictions[i]][1]] +=1
    return result
    
## CAH

# Dans cette partie, la classification ascendante hiérarchique a été implémentée.
# Les fonctions prennent en argument le nombre de clusters désirés

# dendrogram_h affiche le dendrogramme de la CAH avec 10 clusters représentés
# hist_clusters_h affiche l'histogramme des classes créées par le clustering
# silhouette_h renvoit la silhouette du clustering
# confusion_matrix_test_h affiche les labels des classes et renvoit la matrice de confusion de la base de donnée de test du clustering
# hist_clusters_h_sl affiche l'histogramme des classes créées par le clustering avec la distance Single Linkage 

digits_train_no_class = []

for i in digits_train:
    digits_train_no_class.append(i[:64])
    
def clustering_h(n_clusters):
    clustering_hierarchique = linkage(digits_train_no_class,method='ward')
    return fcluster(clustering_hierarchique,t=n_clusters,criterion="maxclust")

def dendrogram_h():
    clustering_hierarchique = linkage(digits_train_no_class,method='ward')
    dendrogram(clustering_hierarchique,truncate_mode="level",p=5,color_threshold=400)
    plt.show()

def get_clusters_index_h(fclusters,n_clusters):
    clusters_index = [[] for i in range(n_clusters)]
    for i in range(len(fclusters)):
        clusters_index[fclusters[i]-1].append(i)
    return clusters_index
    
def hist_clusters_h(n_clusters):
    fclusters = clustering_h(n_clusters)
    cluster_classes = [[] for i in range(n_clusters)]
    for i in range(len(fclusters)):
        cluster_classes[fclusters[i]-1].append(digits_train[i][64])
    clusters_classes_label = []
    for i in cluster_classes:
        clusters_classes_label.append([i,np.argmax(np.bincount(i))])
    clusters_classes_label.sort(key = lambda t : t[1])
    fig = plt.figure()
    for i in range(len(clusters_classes_label)):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.hist(clusters_classes_label[i][0],rwidth = 0.1,bins = np.arange(11)-0.5)
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
        ax.text(0,7,str(clusters_classes_label[i][1]))
    plt.show()
    
def silhouette_h(n_clusters):
    fclusters = clustering_h(n_clusters)
    return silhouette_score(digits_train,fclusters)

def compute_centers_h(fclusters,n_clusters,clusters_index):
    clusters = [ [digits_train[i][:64] for i in clusters_index[j]] for j in range(len(clusters_index))]
    centers = [np.mean(i,axis = 0) for i in clusters]
    return centers


def confusion_matrix_test_h(n_clusters):
    fclusters = clustering_h(n_clusters)
    clusters_index = get_clusters_index_h(fclusters,n_clusters)
    centers = compute_centers_h(fclusters,n_clusters,clusters_index)
    return confusion_matrix_test_i(clusters_index,centers)

def hist_clusters_h_sl(n_clusters):
    clustering_hierarchique = linkage(digits_train_no_class,method='single')
    fclusters = fcluster(clustering_hierarchique,t=n_clusters,criterion="maxclust")
    cluster_classes = [[] for i in range(n_clusters)]
    for i in range(len(fclusters)):
        cluster_classes[fclusters[i]-1].append(digits_train[i][64])
    clusters_classes_label = []
    for i in cluster_classes:
        clusters_classes_label.append([i,np.argmax(np.bincount(i))])
    clusters_classes_label.sort(key = lambda t : t[1])
    fig = plt.figure()
    for i in range(len(clusters_classes_label)):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.hist(clusters_classes_label[i][0],rwidth = 0.1,bins = np.arange(11)-0.5)
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
        ax.text(0,7,str(clusters_classes_label[i][1]))
    plt.show()