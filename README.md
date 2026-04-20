# Neural-Surrogate-for-2D-Heat-Transfer

> **Objectif** : Entraîner un réseau de neurones convolutif (CNN) à prédire la distribution de température finale d'une plaque 2D à partir de ses conditions aux limites (sources de chaleur), afin de remplacer un solveur itératif lent (différences finies).  
> Ce README présente un **plan détaillé** du projet, suivi du **contenu des 1/3 du plan** (sections 1 à 4).

---

## 📋 Plan détaillé (structure complète du projet)

| Section | Titre |
|---------|-------|
| 1 | Contexte et objectif |
| 2 | Description du problème physique |
| 3 | Génération des données (solveur numérique) |
| 4 | Architecture du réseau de neurones (CNN) |
| 5 | Préparation des données (augmentation, normalisation) |
| 6 | Entraînement (loss, métriques, optimiseur) |
| 7 | Évaluation et comparaison avec le solveur traditionnel |
| 8 | Résultats attendus |
| 9 | Utilisation du modèle (inférence rapide) |
| 10 | Perspectives et améliorations |

---

## 📘 Contenu des 1/3 du plan (sections 1 à 4)

### 1. Contexte et objectif

Dans de nombreux problèmes d’ingénierie, la simulation de la diffusion de chaleur dans une plaque 2D nécessite la résolution itérative de l’équation de la chaleur. Les méthodes classiques (différences finies, éléments finis) sont précises mais coûteuses en temps de calcul, surtout lorsqu’il faut explorer de nombreuses configurations (optimisation, contrôle en temps réel).

**Objectif principal** :  
Construire un **surrogate model** basé sur un CNN capable de prédire instantanément le champ de température stationnaire à partir de la carte des sources de chaleur et des conditions aux limites.

**Avantages attendus** :
- Réduction du temps d’inférence de plusieurs secondes (solveur) à quelques millisecondes (CNN).
- Possibilité d’intégration dans une boucle d’optimisation ou un contrôle temps réel.
- Généralisation à des motifs de sources non vus pendant l’entraînement.

### 2. Description du problème physique

#### 2.1 Équation de la chaleur 2D stationnaire

Pour une plaque isotrope homogène, sans convection ni rayonnement, la température \( T(x,y) \) satisfait :

\[
\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = -\frac{q(x,y)}{k}
\]

où :
- \( q(x,y) \) : source de chaleur volumique (W/m³)
- \( k \) : conductivité thermique (W/(m·K))

#### 2.2 Domaine et conditions aux limites

- Domaine : rectangle \([0, L_x] \times [0, L_y]\) discrétisé en grille \(N_x \times N_y\).
- Conditions aux limites : 
  - Température imposée (Dirichlet) sur certains bords (par ex. \(T=300\) K à gauche).
  - Flux nul (Neumann) sur les autres bords (isolant parfait).

#### 2.3 Entrées / sorties du surrogate

- **Entrée** (carte 2D) : distribution des sources de chaleur \( q(x,y) \) (peut être nulle par endroits, positive ou négative).
- **Sortie** (carte 2D) : champ de température stationnaire \( T(x,y) \).

Le CNN doit donc apprendre l’opérateur \( \mathcal{F} : q \mapsto T \).

### 3. Génération des données (solveur numérique)

#### 3.1 Solveur itératif (référence)

On utilise un schéma de différences finies centrées avec relaxation de Gauss‑Seidel.

**Algorithme** :
1. Initialisation de \( T \) (ex: 300 K partout).
2. Pour chaque itération, mettre à jour chaque nœud intérieur :
   \[
   T_{i,j}^{\text{new}} = \frac{1}{4}\left(T_{i-1,j} + T_{i+1,j} + T_{i,j-1} + T_{i,j+1} + \frac{\Delta x^2}{k} q_{i,j}\right)
   \]
3. Appliquer les conditions aux limites.
4. Répéter jusqu’à convergence (\( \max|T^{\text{new}} - T| < \epsilon \)).

#### 3.2 Création du dataset

On génère **10 000 paires** \((q, T)\) en variant aléatoirement :
- Nombre, positions et intensités des sources (points chauds/froids).
- Possibilité d’ajouter un bruit de fond (petites variations).

**Paramètres de discrétisation** :
- Grille : \(64 \times 64\) (compromis précision / mémoire).
- Conductivité \(k = 1\) (normalisée).
- Seuil de convergence : \(10^{-5}\).

**Exemple de code Python (pseudo)** :
```python
import numpy as np
from scipy.ndimage import gaussian_filter

def generate_random_source(shape=(64,64), n_sources=3):
    q = np.zeros(shape)
    for _ in range(n_sources):
        x, y = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
        intensity = np.random.uniform(-100, 500)
        q[x, y] += intensity
    # Lissage pour créer des sources étendues réalistes
    q = gaussian_filter(q, sigma=1.5)
    return q

def solve_heat_equation(q, max_iter=20000, tol=1e-5):
    T = np.ones_like(q) * 300   # initial
    # ... (implémentation du solveur Gauss-Seidel)
    return T
