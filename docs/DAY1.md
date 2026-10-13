# Labo guidé – Jour 1

## 1. Données dans `data/`

Les tables de la compétition sont sur git, dans le dossier `data/`.
Vous devriez voir `X_train.csv`, `y_train.csv`, `X_test.csv` et `sample_submission.csv`. Chaque membre de l'équipe a besoin de ces fichiers CSV en local.

### Sujet des données et objectif

- **Contexte** : La maladie de Parkinson (MP) provoque des symptômes moteurs (tremblements, rigidité, bradykinésie) évalués via le score MDS-UPDRS (0–132). Les patients prenant de la lévodopa oscillent entre un état **ON** (symptômes soulagés) et un état **OFF** (les effets du médicament se sont dissipés). Le score OFF reflète la neurodégénérescence sous-jacente, mais les mesures cliniques sont biaisées par le moment de la prise des médicaments, la subjectivité humaine de l'évaluation et les données manquantes (les tests OFF sont inconfortables et souvent ignorés).
- **Les données** : Visites synthétiques de patients de cohortes multiples à des intervalles irréguliers. Les colonnes incluent les données démographiques, génétiques, la dose de médicament (`ledd`), les scores moteurs mesurés `on` et `off`, et les délais de prise (`time_since_intake_`*). Les patients du jeu de test ne chevauchent pas ceux de l'entraînement (séparation par `patient_id`).
- **L'objectif** : Prédire le **véritable score OFF** sans biais (`target`) pour chaque visite de patient en capturant la progression de la maladie au fil du temps, en exploitant les délais de prise des médicaments, et en traitant les valeurs manquantes comme un signal. Voir [CONTEXT.md](CONTEXT.md) pour le contexte clinique complet.

——

## 2. Explorer les données (EDA)

L'**Analyse Exploratoire des Données (EDA)** consiste à observer les tables *avant* de choisir un modèle : dimensions, valeurs manquantes, groupes de `patient_id`, ce qui ressemble à une fuite de données (leak). Dans Bob, c'est l'étape **G-EDA** (`explore-ml-data`). Elle génère `data/eda.py`, un court rapport narratif `data/eda.md` et des pages interactives `data/eda_<table>.html`.

### Exemple de prompt :

```
Explore mes données dans le dossier `data/`.
```

### Questions du cahier d'exercices

Questions auxquelles réfléchir pendant que vous parcourez les pages d'EDA (`data/eda_<table>.html`, ou le rapport `eda` sur Skore Hub). Inutile de rédiger des réponses. Elles sont là pour vous indiquer ce qu'il faut observer.

1. Combien de visites et combien de patients distincts y a-t-il dans l'entraînement ? Dans le test ? Quel est le nombre typique de visites par patient, et quel est le maximum ?
2. Quelles sont les trois colonnes ayant le plus de valeurs manquantes ? Quelle fraction des visites n'a **aucun** score `off` ? **Aucun** score `on` ? Y a-t-il une visite où les deux sont manquants ?
3. Observez la distribution de `target`. Quelle plage couvre-t-elle par rapport à la plage théorique de 0–132 du MDS-UPDRS ? Est-elle asymétrique ?
4. Quelles colonnes semblent les plus associées à `target` ? Le score `off` (le score mesuré) est-il une copie presque exacte de `target`, ou est-il clairement différent ? Qu'en est-il de `on` ?
5. La variable `ledd` est-elle manquante de façon aléatoire, ou est-elle manquante plus souvent pour certaines valeurs de `cohort` ? Que cela pourrait-il signifier d'un point de vue clinique ?
6. Quelles colonnes ne fourniriez-vous **pas** à un modèle, et pourquoi ? (Indice : pensez aux identifiants et aux fuites de données).

## 3. Dummy mean (moyenne naïve) : un seuil de base

Avant de construire un vrai modèle, construisez le plus paresseux possible. Un modèle **dummy mean** (régresseur naïf par la moyenne) ignore tout ce qu'il sait d'une visite (âge, médicaments, scores mesurés) et donne toujours la même réponse : la `target` **moyenne** sur toutes les visites d'entraînement. Si le véritable score OFF moyen dans l'entraînement est de 30, il prédira 30 pour chaque patient, à chaque visite.
Cela semble inutile, et en tant que modèle, ça l'est. En tant que **point de référence** (*baseline*), c'est essentiel :

- **Il fixe le seuil de base.** Son erreur vous indique à quel point vous avez tort quand vous ne savez rien. Tout modèle réel doit faire mieux ; sinon, ses colonnes ne sont d'aucune utilité.
- **Il détecte les bugs tôt.** Si un modèle "intelligent" obtient un score identique ou pire que le modèle naïf, c'est que quelque chose en amont est cassé, et non le modèle : les données ont été mal chargées, les lignes ont été désalignées de leurs cibles, la métrique est mal calculée, ou le fichier de soumission est mal formaté.
- **Il donne du sens à vos scores.** Une RMSE de 8 ne veut rien dire en soi. "8, comparé à 12 pour le modèle naïf" signifie que votre modèle supprime un tiers de l'erreur.

Commencez par charger les visites.

Chaque **ligne** correspond à une **visite**. La cible est un **véritable score OFF sans biais** MDS-UPDRS (`y_train.target`). Les scores cliniques ON/OFF sont biaisés (subjectivité, valeurs manquantes, timing de la lévodopa).

- `patient_id` : plusieurs visites par patient. La séparation (holdout) Kaggle se fait **par patient**.
- **Valeurs manquantes** : `on`, `off`, `ledd`, `gene`, et les délais de prise sont souvent manquants.
- **ON / OFF / LEDD / timing** : `on`, `off`, `ledd`, `time_since_intake_on`, `time_since_intake_off`. Dans l'entraînement, le nombre d'années depuis le diagnostic est `age - age_at_diagnosis` (le test peut déjà comporter `time_since_diagnosis`).

```python
import pandas as pd

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index")
y = visits["target"]
```

Utilisez `sample_submission.csv` comme modèle de structure : colonnes `Index,target`. Chaque expérience écrit son fichier dans `submissions/<report-key>.csv`, de sorte que les colonnes et le nom de fichier correspondent à la compétition.

- **Ce que c'est** : Le modèle naïf (Dummy) prédit toujours le même nombre : le véritable score OFF moyen de l'entraînement.
- **Comment ça marche** : il ignore toutes les colonnes. Vous passez quand même un `X` pour que `skore.evaluate` puisse aligner les mêmes lignes avec `y`.
- **Pourquoi le faire ici** : si vous n'arrivez pas à battre ce seuil de base, c'est que le chargement des données, la métrique ou l'upload est cassé - pas le modèle.

```python
from sklearn.dummy import DummyRegressor
from skore import evaluate

feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
X = visits[feature_cols]
y = visits["target"]

dummy = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)
report.metrics.rmse()
```

`skore.evaluate` est le point d'entrée. Sans `splitter=`, il utilise par défaut `splitter=0.2` (une retenue aléatoire de lignes). C'est suffisant pour vérifier le seuil de base ; les séparations groupées (grouped splits) viendront le Jour 2.

**Project** stocke les rapports. **Hub** est requis pour une soumission valide (l'URL affichée après `put`). `name=` est votre projet à l'intérieur de l'espace de travail (workspace) du hub depuis `.skore`. Faites toujours `load_skore_credentials()` puis `login(mode="hub")` :

```python
from parkinson.hub import load_skore_credentials
from skore import Project, login

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)
# La console affiche : Consult your report at [https://skore.probabl.ai/](https://skore.probabl.ai/)…
```

`Project.get` se fait par **id** depuis `project.summarize()`, et non par la clé sous forme de chaîne de caractères que vous avez passée à `put`.

### Exemple de prompt :

```
Exécute un "dummy regressor" de sklearn pour servir de modèle de référence. Pousse-le sur le hub avec le nom '01_dummy' comme nom de rapport vers le projet 'bobathon-esilv'.
```

### Questions du cahier d'exercices

Ouvrez l'URL du rapport `01_dummy` sur Skore Hub et répondez à partir de là.

1. Quelle est la RMSE du modèle naïf (dummy) ? Comparez-la avec l'écart-type (standard deviation) de `target` issu de votre EDA. Pourquoi ces deux nombres sont-ils si proches ?
2. Regardez le graphique d'erreur de prédiction (prédit vs réel). Décrivez sa forme. À quoi ressemblerait un modèle parfait sur ce même graphique ?
3. Notez la RMSE du modèle naïf quelque part de visible. Chaque modèle à partir de maintenant devra la battre. De combien un modèle devrait-il la battre avant que vous ne considériez l'amélioration comme réelle, plutôt que comme du bruit ?

