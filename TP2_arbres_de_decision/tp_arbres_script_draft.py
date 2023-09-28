#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

from sklearn import tree, datasets
from tp_arbres_source import (rand_gauss, rand_bi_gauss, rand_tri_gauss,
                              rand_checkers, rand_clown,
                              plot_2d, frontiere)


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 6,
          'font.size': 12,
          'legend.fontsize': 12,
          'text.usetex': False,
          'figure.figsize': (10, 12)}
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")
_ = sns.axes_style()

############################################################################
# Data Generation: example
############################################################################

np.random.seed(1)

n = 100
mu = [1., 1.]
sigma = [1., 1.]
rand_gauss(n, mu, sigma)


n1 = 20
n2 = 20
mu1 = [1., 1.]
mu2 = [-1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
data1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

n1 = 50
n2 = 50
n3 = 50
mu1 = [1., 1.]
mu2 = [-1., -1.]
mu3 = [1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
sigma3 = [0.9, 0.9]
data2 = rand_tri_gauss(n1, n2, n3, mu1, mu2, mu3, sigma1, sigma2, sigma3)

n1 = 50
n2 = 50
sigma1 = 1.
sigma2 = 5.
data3 = rand_clown(n1, n2, sigma1, sigma2)


n1 = 114  # XXX : change
n2 = 114
n3 = 114
n4 = 114
sigma = 0.1
data4 = rand_checkers(n1, n2, n3, n4, sigma)

#%%
############################################################################
# Displaying labeled data
############################################################################

plt.close("all")
plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(141)
plt.title('First data set')
plot_2d(data1[:, :2], data1[:, 2], w=None)

plt.subplot(142)
plt.title('Second data set')
plot_2d(data2[:, :2], data2[:, 2], w=None)

plt.subplot(143)
plt.title('Third data set')
plot_2d(data3[:, :2], data3[:, 2], w=None)

plt.subplot(144)
plt.title('Fourth data set')
plot_2d(data4[:, :2], data4[:, 2], w=None)

#%%
############################################
# ARBRES
############################################


# Q2. Créer deux objets 'arbre de décision' en spécifiant le critère de
# classification comme l'indice de gini ou l'entropie, avec la
# fonction 'DecisionTreeClassifier' du module 'tree'.

# Construction des classifieur
dt_entropy = tree.DecisionTreeClassifier(criterion='entropy')
dt_gini = tree.DecisionTreeClassifier(criterion='gini')

# Simulation de l'échantillon
n = 456
data = rand_checkers(n1=n//4, n2=n//4, n3=n//4, n4=n//4)
n_samples = len(data)
X = data[:,:2]
Y = np.asarray(data[:,-1], dtype=int)

# Entraînement des deux modèles
dt_gini.fit(X, Y)
dt_entropy.fit(X, Y)

print("Gini criterion")
print(dt_gini.get_params())
print(dt_gini.score(X, Y))

print("Entropy criterion")
print(dt_entropy.get_params())
print(dt_entropy.score(X, Y))

#%%
# Afficher les scores en fonction du paramètre max_depth
# Initialisation 
dmax = 12      # choix arbitraire   
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)
plt.figure(figsize=(15, 10))

# Boucle principale
for i in range(dmax):
    # Critère : entropie
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', 
                                             max_depth=i+1)
    dt_entropy.fit(X,Y)
    scores_entropy[i] = dt_entropy.score(X, Y)

    # Critère : indice de Gini
    dt_gini = tree.DecisionTreeClassifier(criterion='gini', 
                                          max_depth=i+1)
    dt_gini.fit(X,Y)
    scores_gini[i] = dt_gini.score(X,Y)

    # Affichage progressif des frontières en fonction de la profondeur de l'arbre
    plt.subplot(3, 4, i + 1)
    frontiere(lambda x: dt_gini.predict(x.reshape((1, -1))), X, Y, step=50, samples=False)
plt.draw()

plt.figure()
plt.plot(1-scores_entropy, label="entropy")
plt.plot(1-scores_gini, label="gini")
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
plt.draw()

print("Scores with entropy criterion: ", scores_entropy)
print("Scores with Gini criterion: ", scores_gini)

#%%
# Q3 Afficher la classification obtenue en utilisant la profondeur qui minimise le pourcentage d’erreurs
# obtenues avec l’entropie

dt_entropy.max_depth = np.argmin(1-scores_entropy)+1
plt.figure(figsize=(6,3.2))
frontiere(lambda x: dt_entropy.predict(x.reshape((1, -1))), X, Y, step=100)
plt.draw()
print("Best scores with entropy criterion: ", dt_entropy.score(X, Y))

#%%
# Q4.  Exporter la représentation graphique de l'arbre: Need graphviz installed
# Voir https://scikit-learn.org/stable/modules/tree.html#classification
import graphviz
tree.plot_tree(dt_entropy)
data = tree.export_graphviz(dt_entropy)
graph = graphviz.Source(data)
graph.render('./graphviz/dt_entropy', format='pdf')


#%% Arbre plus petit

little_t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
little_t.fit(X,Y)
data_little_t = tree.export_graphviz(little_t)
graph_lt = graphviz.Source(data_little_t)
graph_lt.render('./graphviz/example', format='pdf')

#%%
# Q5 :  Génération d'une base de test

# Initialisation de l'échantillon de test
data_test = rand_checkers(n1=40, n2=40, n3=40, n4=40)
X_test = data_test[:,:2]
Y_test = np.asarray(data_test[:,-1], dtype=int)

# Initialisation
dmax = 30                             # choisi arbitrairement
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)

# Boucle principale
for i in range(dmax):
    # Critère : entropie
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', 
                                             max_depth=i+1)
    dt_entropy.fit(X,Y)
    scores_entropy[i] = dt_entropy.score(X_test, Y_test)

    # Critère : indice de Gini
    dt_gini = tree.DecisionTreeClassifier(criterion='gini', 
                                          max_depth=i+1)
    dt_gini.fit(X,Y)
    scores_gini[i] = dt_gini.score(X_test,Y_test)

# Affichage des courbes
plt.figure(figsize=(6,3.2))
plt.plot(1-scores_entropy)
plt.plot(1-scores_gini)
plt.legend(['entropy', 'gini'])
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
plt.title("Testing error")

print(scores_entropy)

#%%
# Q6. même question avec les données de reconnaissances de texte 'digits'
from sklearn import model_selection

# Importation et construction des échantillons test/train
digits = datasets.load_digits()
n_samples = len(digits.data)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(digits.data,
                                                                    digits.target, 
                                                                    test_size=0.8)

# Initialisation
dmax = 15
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)

# Boucle principale
for i in range(dmax):
    # Critère : entropie
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', 
                                             max_depth=i+1)
    dt_entropy.fit(X_train,Y_train)
    scores_entropy[i] = dt_entropy.score(X_train, Y_train)

    # Critère : indice de Gini
    dt_gini = tree.DecisionTreeClassifier(criterion='gini', 
                                          max_depth=i+1)
    dt_gini.fit(X_train,Y_train)
    scores_gini[i] = dt_gini.score(X_train,Y_train)

# Affichage des courbes
plt.figure(figsize=(6,3.2))
plt.plot(1-scores_entropy, label="entropy")
plt.plot(1-scores_gini, label="gini")
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
plt.draw()

print("Scores with entropy criterion: ", scores_entropy)
print("Scores with Gini criterion: ", scores_gini)


#%%
# Initialisation
dmax = 15
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)

# Boucle principale
for i in range(dmax):
    # Critère : entropie
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', 
                                             max_depth=i+1)
    dt_entropy.fit(X_train,Y_train)
    scores_entropy[i] = dt_entropy.score(X_test, Y_test)

    # Critère : indice de Gini
    dt_gini = tree.DecisionTreeClassifier(criterion='gini', 
                                          max_depth=i+1)
    dt_gini.fit(X_train,Y_train)
    scores_gini[i] = dt_gini.score(X_test,Y_test)

# Affichage des courbes
plt.figure(figsize=(6,3.2))
plt.plot(1-scores_entropy, label="entropy")
plt.plot(1-scores_gini, label="gini")
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
plt.draw()

print("Scores with entropy criterion: ", scores_entropy)
print("Scores with Gini criterion: ", scores_gini)

#%%
# Erreurs pour d'autres partitionnements

# on fixe de nouveaux découpage de notre échantillon
X_a1, X_t1, Y_a1, Y_t1 = model_selection.train_test_split(digits.data,
                                                          digits.target, 
                                                          test_size=0.8,
                                                          random_state=21)
X_a2, X_t2, Y_a2, Y_t2 = model_selection.train_test_split(digits.data,
                                                          digits.target, 
                                                          test_size=0.8,
                                                          random_state=31)                        
dmax = 15
scores_entropy1 = np.zeros(dmax)
scores_gini1 = np.zeros(dmax)
scores_entropy2 = np.zeros(dmax)
scores_gini2 = np.zeros(dmax)

for i in range(dmax):
    # Premier découpage
    dt_entropy1 = tree.DecisionTreeClassifier(criterion='entropy', 
                                             max_depth=i+1)
    dt_entropy1.fit(X_a1,Y_a1)
    scores_entropy1[i] = dt_entropy1.score(X_t1, Y_t1)

    dt_gini1 = tree.DecisionTreeClassifier(criterion='gini', 
                                          max_depth=i+1)
    dt_gini1.fit(X_a1,Y_a1)
    scores_gini1[i] = dt_gini1.score(X_t1,Y_t1)

    # Deuxième découpage 
    dt_entropy2 = tree.DecisionTreeClassifier(criterion='entropy', 
                                             max_depth=i+1)
    dt_entropy2.fit(X_a2,Y_a2)
    scores_entropy2[i] = dt_entropy2.score(X_t2, Y_t2)

    dt_gini2 = tree.DecisionTreeClassifier(criterion='gini', 
                                          max_depth=i+1)
    dt_gini2.fit(X_a2,Y_a2)
    scores_gini2[i] = dt_gini2.score(X_t2,Y_t2)


# Affichage des deux courbes
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(8, 4))

# Première partition
ax1.plot(1 - scores_entropy1, label="entropy")
ax1.plot(1 - scores_gini1, label="gini")
plt.xlabel('Profondeur maximale', fontsize=12)
plt.ylabel("Taux d'erreur", fontsize=12)

# Deuxième partition
ax2.plot(1 - scores_entropy2, label="entropy")
ax2.plot(1 - scores_gini2, label="gini")
ax2.legend()
plt.xlabel('Profondeur maximale', fontsize=12)
plt.ylabel("Taux d'erreur", fontsize=12)

# Afficher la figure
fig.suptitle("Courbes d'erreur avec différentes coupes", fontsize=15)
plt.show()


#%%
# Q7. estimer la meilleur profondeur avec un cross_val_score
from sklearn.model_selection import cross_val_score

# Initialisation avec graine fixée
np.random.seed(123)
dmax = 30
X = digits.data
Y = digits.target
error = np.zeros(dmax)

# Boucle de calcul de l'erreur
for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i+1)
    error[i] = np.mean(1-cross_val_score(dt_entropy, X, Y, cv=5))

# Affichage de la courbe
plt.figure(figsize=(6,3.2))
plt.plot(error)
best_depth = np.argmin(error)+1
print("Best depth: ", best_depth)

#%%
# Exportation de meilleur arbre
dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8)
dt.fit(X_train,Y_train)
data_f = tree.export_graphviz(dt)
graph_lt = graphviz.Source(data_f)
graph_lt.render('./graphviz/final_tree', format='pdf')

#%%
from sklearn.model_selection import learning_curve

# Calcul des courbes d'apprentissage
dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8)
n_samples, train_curve, test_curve = learning_curve(dt, X, Y,train_sizes=np.linspace(0.1, 1, 8))

# Mise en place de l'affichage
plt.figure(figsize=(6,3.2))
plt.grid()

# colorisation des intervalles de confiance
plt.fill_between(n_samples, np.mean(train_curve, axis=1) -1.96*np.std(train_curve, axis=1),
                  np.mean(train_curve, axis=1) + 1.96*np.std(train_curve, axis=1), alpha=0.1)
plt.fill_between(n_samples, np.mean(test_curve, axis=1) -1.96*np.std(test_curve, axis=1),
                  np.mean(test_curve, axis=1) + 1.96*np.std(test_curve, axis=1), alpha=0.1)

# Affichage des courbes
plt.plot(n_samples, np.mean(train_curve, axis=1),"o-", label="train")
plt.plot(n_samples, np.mean(test_curve, axis=1), "o-", label="test")
plt.legend(loc="lower right")
plt.xlabel("Size of sample in the training set")
plt.ylabel("Accuracy")
plt.title("Learning curve for best decision tree")
