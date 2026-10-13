# Labo guidé – Jour 2

Lors du Jour 1, vous avez chargé les données de la compétition, vous les avez explorées (EDA) pour voir ce qui manque et comment les visites se regroupent par patient, et vous avez poussé un modèle de référence naïf par la moyenne (*dummy mean baseline*) sur Skore Hub comme le score que tout modèle doit battre. Aujourd'hui, vous allez construire de vrais modèles, apprendre à les évaluer honnêtement avec une validation croisée groupée par patient, et les soumettre sur Kaggle.

## 1. Un modèle linéaire simple

Les variables cliniques numériques (`age`, `on`, `off`, `ledd`, les délais) peuvent porter un signal additif simple.

- **Ce que c'est** : Ridge trace une ligne droite : prédiction du véritable OFF ≈ ordonnée à l'origine (*intercept*) + (un poids × chaque colonne).
- **Comment ça marche** : il choisit les poids pour que les prédictions correspondent à `y`, puis `alpha` tire ces poids vers zéro pour qu'une seule colonne bruitée ne puisse pas dominer.
- **Pourquoi le faire ici** : une première vérification pour s'assurer que les nombres ont une quelconque importance. Ridge ne consomme que des nombres sans trous (valeurs manquantes), donc imputez d'abord par la médiane et laissez de côté les chaînes de caractères comme `gene` pour plus tard.

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from skore import evaluate

ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)
report = evaluate(ridge, X, y)
```

`make_pipeline(*steps)` assemble les transformateurs puis le régresseur. Comparez le modèle naïf et le modèle ridge dans un seul rapport :

```python
report = evaluate(
    {"dummy": dummy, "ridge": ridge},
    X,
    y,
)
```

Faites un `put` avec une nouvelle clé. La retenue de lignes aléatoire par défaut peut toujours faire "fuiter" (*leak*) le même patient des deux côtés - la validation croisée groupée est l'étape 3.

### Exemple de prompt :

```
Exécute un modèle linéaire et pousse le rapport sur le hub. Merci de comparer les résultats avec le modèle naïf (dummy).
```

### Questions du cahier d'exercices

Ouvrez le rapport de comparaison sur Skore Hub.

1. De quel **pourcentage** Ridge réduit-il la RMSE par rapport au modèle naïf ? Est-ce une grande victoire ou une petite ?
2. Regardez les coefficients du modèle Ridge. Quelle variable (*feature*) a le poids le plus important en valeur absolue ? Le signe des coefficients `off` et `on` correspond-il à ce que vous attendiez cliniquement ?
3. Les variables n'ont pas été mises à l'échelle (*scaled*). Pourquoi cela signifie-t-il que vous ne **pouvez pas** lire "`age` compte plus que `ledd`" directement à partir de la taille brute des coefficients ?

## 2. Ajuster Ridge et soumettre sur Kaggle

`Ridge(alpha=…)` est le bouton de réglage : un `alpha` plus grand réduit davantage les coefficients. Modifiez-le, relancez `evaluate`, gardez celui qui bat le modèle naïf avec la plus grande marge.

```python
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=10.0),  # essayez 0.1, 1.0, 10.0, …
)
report = evaluate(ridge, X, y)
report.metrics.rmse()
project.put("02_ridge", report)
```

`evaluate` n'évalue que sur l'entraînement. Le fichier que vous uploadez correspond à ce pipeline **entraîné (*fit*) sur toutes les visites d'entraînement**, puis utilisé pour prédire (`predict`) sur `X_test` (les mêmes colonnes que `X`) :

```python
from pathlib import Path
from sklearn.base import clone

final = clone(ridge).fit(X, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])

submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "02_ridge.csv", index=False)
```

Uploadez `submissions/02_ridge.csv` sur Kaggle. Le fichier porte le nom de la clé du rapport, de sorte que chaque expérience conserve son propre CSV - un modèle ultérieur n'écrasera jamais une soumission précédente. Une soumission n'est valide que si :

1. Le code a été écrit et exécuté dans **Bob**, l'évaluation via **skore**.
2. Vous avez "poussé" (`put`) ce rapport Ridge ; la console affiche `https://skore.probabl.ai/…`.
3. L'en-tête du CSV est `Index,target`, avec une ligne par visite de test.
4. La **Description de la soumission** (Submission Description) Kaggle contient cette URL de rapport (sinon la ligne est invalide, même si Kaggle a noté le fichier).

Les modèles ultérieurs utiliseront le même chemin entraînement (*fit*) → CSV → URL.

### Questions du cahier d'exercices

1. Parmi les valeurs de `alpha` que vous avez essayées, de combien la RMSE varie-t-elle réellement ? Pourquoi `alpha` pourrait-il avoir si peu d'importance avec seulement quelques variables et des milliers de lignes ?
2. Les coefficients ont-ils beaucoup changé entre le plus petit et le plus grand `alpha` ? Comparez-les sur le hub.

## 3. Évaluer avec une validation croisée (CV) groupée par patient

Le paramètre par défaut `splitter=0.2` mélange les **lignes**. C'est trop indulgent ici.

- **Ce que c'est** : GroupKFold est une répétition générale de la séparation Kaggle : un patient entier va soit dans l'entraînement, soit dans le pli de validation (*held-out*), jamais dans les deux.
- **Comment ça marche** : il divise les patients en 5 groupes. À 5 reprises, il s'entraîne sur 4 groupes et évalue sur le 5ème, puis vous lisez la RMSE moyenne.
- **Pourquoi le faire ici** : la même personne a de nombreuses visites. Une séparation aléatoire des lignes est une **fuite de données (leakage)** : des visites d'un même patient se retrouvent des deux côtés, le modèle mémorise donc cette personne au lieu de généraliser, et la métrique semble meilleure que lors de la compétition Kaggle (qui retient des patients entiers). La CV groupée bloque cette fuite.

Le chemin sklearn de skore appelle `splitter.split(X, y)` **sans** `groups=`, donc précalculez les paires d'indices :

```python
from sklearn.model_selection import GroupKFold
from skore import evaluate

groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
report = evaluate(
    {"dummy": dummy, "ridge": ridge},
    X,
    y,
    splitter=cv_splits,
)
report.metrics.summarize()          # tableau des métriques
report.metrics.get("rmse")          # ou report.metrics.rmse()
```

- `splitter=` une liste de paires d'indices (ou `5`, ou un "CV splitter") → `CrossValidationReport`
- plusieurs estimateurs (liste ou dictionnaire) → `ComparisonReport`

Utilisez `cv_splits` à partir de maintenant. Faites un `put` de ce rapport avec une nouvelle clé.

DataOps de skrub peut attacher des `groups` sur le graphe - voir l'étape 6.

### Questions du cahier d'exercices

Ouvrez le rapport de validation croisée sur Skore Hub.

1. Comparez la RMSE de la CV groupée avec la RMSE de la séparation aléatoire du modèle naïf du Jour 1 et des étapes 1-2. Le score de quel modèle a le plus changé ?
2. Laquelle des deux estimations (séparation aléatoire ou CV groupée) est la plus proche de votre score public Kaggle ? Qu'est-ce que cela vous apprend sur celle à laquelle vous devez faire confiance à partir de maintenant ?

## 4. HistGradientBoosting : les valeurs manquantes comme signal

ON/OFF/LEDD/génétique sont souvent manquants ; le fait d'être manquant fait partie du processus de génération (voir [CONTEXT.md](CONTEXT.md)). Ridge a dû *remplir* ces trous avec une médiane. Ici, nous gardons les trous.

- **Ce que c'est** : un **arbre de décision** est un organigramme de questions oui/non (`age > 62 ?`, `on` est manquant ?). Chaque visite descend dans l'organigramme jusqu'à une feuille, et cette feuille prédit un nombre (un véritable OFF typique pour les visites qui ont atterri là). **Le Boosting** signifie que l'on ne s'arrête pas à un seul arbre : on fait pousser de nombreux petits arbres à la suite, et chaque nouvel arbre s'entraîne sur les *erreurs* de ceux qui l'ont précédé. La prédiction finale est la somme de toutes ces petites corrections. **Hist** (histogramme) est une astuce d'implémentation : chaque colonne numérique est d'abord découpée en quelques segments (comme un histogramme), de sorte que le modèle examine les identifiants de segments (*buckets*) au lieu de chaque âge distinct. Cela lui permet de rester rapide sur des dizaines de milliers de visites.
- **Comment ça marche** :
  - L'arbre 1 trace une ébauche approximative du véritable OFF.
  - L'arbre 2 examine les résidus (le véritable OFF moins ce que l'arbre 1 a prédit) et tente d'expliquer ce qui est encore faux.
  - Les arbres 3, 4, ... continuent de grignoter l'erreur restante. C'est le "gradient boosting" en termes simples : continuer à ajouter un petit expert sur les erreurs restantes.
  - Lors d'une séparation (*split*), l'algorithme peut envoyer les valeurs **manquantes** à gauche ou à droite de manière délibérée. Il apprend cette route à partir des données d'entraînement - ainsi, "OFF n'a pas été mesuré" peut être un signal, et non un défaut.
  - `random_state=0` ne sert qu'à rendre l'exécution reproductible.
- **Pourquoi le faire ici** : dans cette table, un OFF manquant signifie souvent que la visite était en état ON uniquement (les examens OFF inconfortables sont ignorés). Il s'agit d'une information sur le patient et le protocole, et non d'un bruit à éliminer par imputation. Ne **remplissez pas** les NaN par la médiane, sauf si vous testez cette ablation (le modèle devient-il *moins bon* lorsque vous cachez les trous ?).

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from skore import evaluate

hgbr = HistGradientBoostingRegressor(random_state=0)
report = evaluate(hgbr, X, y, splitter=cv_splits)
```

Les chaînes de caractères posent toujours problème : par défaut, le modèle veut des nombres ou des types `category` de pandas. `categorical_features="from_dtype"` (la valeur par défaut) traite une colonne `category` comme "choisir parmi quelques étiquettes" au lieu d'un faux nombre. Convertissez `cohort` / `gene` avec `.astype("category")` si vous les ajoutez à `X`.

### Questions du cahier d'exercices

Mettez le rapport HGBR à côté de celui de Ridge (mêmes `cv_splits`) et comparez sur Skore Hub.

1. HGBR bat-il Ridge sur **tous** les plis (*folds*), ou seulement en moyenne ?
2. Comparez les temps d'entraînement (*fit*) et de prédiction (*predict*). De combien HGBR est-il plus lent ? Le gain en vaut-il la peine ici ?
3. Regardez l'importance des variables (*permutation importance*). Sur quelles variables HGBR s'appuie-t-il le plus ? Ce classement est-il différent de celui des coefficients Ridge ?

## 5. `tabular_pipeline` de skrub : types mixtes sans encodage manuel

Ridge et le modèle numérique HGBR ci-dessus n'ont jamais vu `gene` ou `cohort` : ce sont des **chaînes de caractères**. Un régresseur sklearn ne peut pas multiplier `"GBA"` par un poids. Quelque chose doit d'abord transformer le texte en nombres. Faire cela à la main (un encodeur personnalisé par colonne) est la raison pour laquelle les pipelines finissent généralement par pourrir.

- **Ce que c'est** : `tabular_pipeline("regressor")` est une recette prête à l'emploi en deux étapes de skrub : (1) `TableVectorizer` transforme une table désordonnée en une matrice numérique, (2) `HistGradientBoostingRegressor` effectue les prédictions. Vous supprimez les identifiants, passez le reste, et le vectoriseur choisit un encodeur **par colonne**.
- **Comment ça marche** : la "cardinalité" désigne le nombre de valeurs distinctes qu'a une colonne.
  - **Faible cardinalité** (une poignée d'étiquettes, par ex. `sexM`, peut-être `cohort`) → **one-hot** : une nouvelle colonne par étiquette, `1` si cette ligne l'a, `0` sinon. Le modèle voit un interrupteur, pas un classement inventé des étiquettes.
  - **Haute cardinalité** (beaucoup de chaînes distinctes, par ex. `gene` si elle est désordonnée) → `StringEncoder` : compresse le texte en quelques dimensions numériques au lieu de centaines de colonnes one-hot. Vous gardez le signal sans faire exploser la largeur de `X`.
  - **Les nombres** (`age`, `on`, `ledd`, …) passent au travers. HGBR peut toujours utiliser leurs NaN, comme à l'étape 4.
  - `Index` et `patient_id` sont des identifiants, pas des variables (*features*) cliniques. Si vous les laissez, le modèle peut mémoriser les identifiants - une autre forme de fuite de données (*leakage*). Supprimez-les (ainsi que la `target`) avant l'entraînement (*fit*).
- **Pourquoi le faire ici** : les colonnes intéressantes de Parkinson sont mixtes : des nombres avec des trous *et* des variables catégorielles. Cette étape vous permet d'éviter de jeter `gene` / `cohort` simplement parce que Ridge ne pouvait pas les lire.

```python
from skrub import tabular_pipeline
from skore import evaluate

# inclure les catégorielles ; garder Index / patient_id hors des variables
X_full = visits.drop(columns=["Index", "patient_id", "target"])
model = tabular_pipeline("regressor")
report = evaluate(model, X_full, y, splitter=cv_splits)
```

Même vectoriseur, votre propre estimateur (si vous voulez ajuster HGBR) :

```python
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor

model = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingRegressor(random_state=0),
)
```

### Questions du cahier d'exercices

1. Comparez ce rapport avec le HGBR uniquement numérique de l'étape 4. L'ajout de `gene` et `cohort` a-t-il fait baisser la RMSE au-delà de la variation entre les plis ?
2. Regardez le pipeline entraîné dans le rapport. Quel encodeur le `TableVectorizer` a-t-il choisi pour chaque colonne de texte ? Ce choix correspond-il à la cardinalité de chaque colonne issue de votre EDA ?

## 6. skrub DataOps : des groupes intégrés dans le graphe

Jusqu'à présent, vous avez jonglé avec des objets séparés : un DataFrame `visits`, une liste `cv_splits`, un pipeline `model`. Facile de s'entraîner sur les mauvaises colonnes ou d'oublier `groups` après un copier-coller. **DataOps** est la manière qu'a skrub d'écrire la *recette* une seule fois, sous forme de graphe, afin que les variables, la cible et la séparation groupée vivent sur le même objet.

- **Ce que c'est** : un DataOp n'est pas encore la prédiction. C'est un plan différé : "quand tu me donneras une table nommée `visits`, supprime `target`, vectorise, puis applique HGBR". `skore.evaluate` parcourt ce plan et exécute la CV (validation croisée) que vous y avez attachée.
- **Comment ça marche** :
  - `skrub.var("visits", visits)` nomme une **entrée**. Plus tard, au moment de la prédiction, vous pourrez passer une table différente sous le même nom (`X_test`).
  - `.skb.mark_as_X(...)` indique à skore "ce nœud est la table des variables". `.skb.mark_as_y()` marque la cible. Ces deux marques sont la façon dont `evaluate` sait ce qu'il faut séparer.
  - Passer `cv=GroupKFold(...)` et `split_kwargs={"groups": groups}` **sur** `mark_as_X` intègre la CV groupée par patient au graphe. Vous ne conservez plus de liste annexe `cv_splits` qui pourrait devenir obsolète.
  - `.skb.apply(transformer)` / `.skb.apply(estimator, y=...)` ajoute une étape, comme `make_pipeline`, mais sur le graphe.
  - `evaluate(pred)` **sans** `splitter=` lit la `cv` / `groups` à partir du DataOp. C'est le but : la séparation (*split*) ne peut pas s'éloigner des données.
  - `skrub.X(value)` est un raccourci pour `skrub.var("X", value).skb.mark_as_X()` : pas de `cv` supplémentaire à moins d'appeler de nouveau `mark_as_X`.
  - `.skb.make_learner()` **fige** le graphe en un `SkrubLearner`. Ensuite, `fit` / `predict` prennent un **dictionnaire d'environnement** (`{"visits": ...}`), car l'entrée était une variable nommée (`var`), et non un simple tableau. Vous en avez besoin pour le fichier Kaggle : le test n'a pas de `target`.
- **Pourquoi le faire ici** : GroupKFold ne fonctionne que si `groups` correspond aux `patient_id` des *mêmes* lignes que `X`. Mettre cela sur le graphe est la manière d'empêcher la fuite de données (*leakage*) de l'étape 3 de revenir sournoisement via un argument oublié.

```python
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skore import evaluate

data = skrub.var("visits", visits)
groups = data["patient_id"]
X_op = data.drop("target", axis=1).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)
y_op = data["target"].skb.mark_as_y()
pred = X_op.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(random_state=0),
    y=y_op,
)

# evaluate lit la cv et les groups depuis le DataOp lorsque le splitter est omis
report = evaluate(pred)
```

### Questions du cahier d'exercices

1. Comparez ce rapport avec celui de l'étape 5. Le modèle est le même ; les scores sont-ils les mêmes ? Si non, expliquez la différence.
2. Quelles colonnes `X_op` contient-elle réellement ? Comparez avec les colonnes supprimées à l'étape 5. Est-ce que `Index` ou `patient_id` apparaissent dans l'importance des variables ? Qu'est-ce que cela signifierait ?

## 7. Entraîner le modèle choisi et soumettre de nouveau

La CV groupée (étapes 3 à 6) vous indique quelle idée est la meilleure. Soumettez-la de la même manière que Ridge (étape 2) : poussez (`put`) un **nouveau** rapport sur le hub, entraînez (`fit`) sur **toutes** les visites d'entraînement, prédisez (`predict`) sur `X_test`, écrivez le CSV nommé d'après cette clé de rapport, et uploadez-le avec l'URL du rapport dans la Description.

```python
from pathlib import Path
from sklearn.base import clone

final = clone(model).fit(X_full, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test.drop(columns=["Index", "patient_id"], errors="ignore"))

submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "<key>.csv", index=False)  # <key> = la clé de rapport utilisée pour put
```

Alignez les colonnes de test avec ce sur quoi vous vous êtes entraîné (mêmes suppressions, mêmes types de données). Pour un DataOp / `SkrubLearner` :

```python
learner = pred.skb.make_learner()
learner.fit({"visits": visits})
pred_test = learner.predict({"visits": X_test})

submission = X_test[["Index"]].copy()
submission["target"] = pred_test
submission.to_csv(submissions_dir / "<key>.csv", index=False)
```

### Questions du cahier d'exercices : Conclusion

Utilisez la vue projet sur Skore Hub pour voir tous les rapports que vous avez poussés (`put`).

1. Quel changement unique (CV groupée, valeurs manquantes conservées, variables catégorielles, changement de modèle) a apporté la plus grande amélioration ? Lequel n'en a apporté aucune ?
2. Choisissez les visites sur lesquelles votre meilleur modèle se trompe le plus (les plus grandes erreurs dans le graphique d'erreur de prédiction). Qu'ont-elles en commun : `off` manquant, `time_since_intake_*` très long, une `cohort` spécifique, début ou fin de maladie ?
3. En vous basant sur [CONTEXT.md](CONTEXT.md), quel problème de modélisation (progression temporelle, timing des médicaments, valeurs manquantes) n'avez-vous **pas** encore abordé ? Quelle variable (*feature*) construiriez-vous ensuite pour y répondre ?
