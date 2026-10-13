<!--
Résumé de l'analyse exploratoire des données pour cet espace de travail, rédigé à partir
de l'exécution de data/eda.py. Fondez chaque affirmation sur ce que l'exécution a réellement
montré - n'inventez pas de faits. Gardez les "Implications pour la modélisation" comme
des suggestions potentielles à évaluer lors de la conception du modèle, et non comme des
décisions finales.
-->

# EDA : Prédiction du score UPDRS (Parkinson)

_Généré à partir de `data/eda.py` le 2025-07-11._

## Le jeu de données en un coup d'œil

- **Tables :** 3 fichiers — `X_train.csv`, `y_train.csv` (fusionnés pour l'EDA), `X_test.csv`
- **Dimensions :** 44 590 lignes × 13 variables (features) + 1 cible (target) (entraînement) ; 11 013 lignes × 13 variables (test)
- **Cible (Target) :** `target` — score moteur UPDRS (Unified Parkinson's Disease Rating Scale) continu ; **tâche de régression**
- **Rapports détaillés :** [eda_train.html](eda_train.html) · [eda_test.html](eda_test.html)

## Observations par colonne

| Colonne | dtype | % nuls | n_unique | Notes |
|---|---|---|---|---|
| `Index` | Int64 | 0 % | 44 590 | Identifiant de ligne — clé de substitution, pas une feature |
| `patient_id` | String | 0 % | 5 576 | Se répète sur plusieurs lignes (plusieurs visites par patient) |
| `cohort` | String | 0 % | 2 | Deux cohortes (A / B) — faible cardinalité |
| `sexM` | Int64 | 0 % | 2 | Binaire : 0 = femme, 1 = homme |
| `gene` | String | **32 %** | 4 | Quatre variants génétiques ; ~1 ligne sur 3 manquante |
| `age_at_diagnosis` | Float64 | **5 %** | 563 | Âge (années) au moment du diagnostic de la maladie de Parkinson |
| `age` | Float64 | 0 % | 1 082 | Âge actuel lors de la visite |
| `ledd` | Float64 | **37 %** | 1 320 | Dose quotidienne équivalente de L-DOPA (mg) |
| `time_since_intake_on` | Float64 | **46 %** | 64 | Heures depuis la dernière dose (état ON) ; très souvent manquant |
| `time_since_intake_off` | Float64 | **79 %** | 178 | Heures depuis la dernière dose (état OFF) ; extrêmement souvent manquant |
| `rater_id` | String | 0 % | 50 | 50 évaluateurs distincts ; biais d'évaluateur potentiel |
| `on` | Float64 | **30 %** | 83 | Score UPDRS dans l'état ON (sous médication) |
| `off` | Float64 | **42 %** | 101 | Score UPDRS dans l'état OFF (sans médication) |

**Points saillants :**
- `time_since_intake_off` est manquant dans 79 % des lignes — la feature la plus incomplète.
- `ledd`, `time_since_intake_on`, `on` et `off` sont chacune manquantes dans 30 à 46 % des lignes.
- `gene` est manquant dans ~32 % des cas — ne peut être ignoré.
- Aucune colonne de type date/heure (*datetime*) détectée (pas de tri temporel par date).
- `rater_id` possède 50 valeurs uniques ; cardinalité modeste mais peut introduire un biais lié à l'évaluateur (*rater-effect bias*).

## Cible (Target)

Score moteur UPDRS continu :

| Stat | Valeur |
|---|---|
| Moyenne | 37.5 |
| Écart-type | 16.5 |
| Min | 0.0 |
| 25e perc. | 25.6 |
| Médiane | 37.3 |
| 75e perc. | 49.3 |
| Max | 109.5 |

Intervalles (bins) de l'histogramme de distribution : `[0, 11)` 2 477 · `[11, 22)` 5 641 · `[22, 33)` 9 764 · `[33, 44)` 10 703 · `[44, 55)` 8 844 · `[55, 66)` 5 203 · `[66, 77)` 1 757 · `[77, 88)` 180 · `[88, 99)` 13 · `[99, 109.5]` 8.

La cible est **globalement en forme de cloche et asymétrique vers la droite** (la queue de distribution au-delà de 77 est très fine, 201 lignes au total). L'espace interquartile (IQR = 23,7 points) couvre une plage clinique significative. Pas d'inflation de zéros. 867 valeurs uniques — effectivement continue.

## Structure

- **Aucune colonne datetime détectée** — pas de colonne d'horodatage explicite.
- **`patient_id` se répète sur plusieurs lignes** (5 576 patients uniques sur 44 590 lignes ≈ 8 visites par patient en moyenne). Il s'agit d'un panel longitudinal classique : plusieurs observations par patient au fil du temps.
- `Index` a un ratio d'unicité (`unique_ratio`) de 1.0 (pur identifiant de ligne — à exclure des variables).
- `age` et `age_at_diagnosis` sont des flottants quasi-continus à forte cardinalité (respectivement 1 082 et 563 valeurs uniques).

## Associations

**Variable (Feature) ↔ Cible (Corrélation de Pearson, triée) :**

| Variable | r de Pearson | V de Cramér |
|---|---|---|
| `off` | **0.871** | 0.406 |
| `on` | **0.669** | 0.368 |
| `ledd` | 0.298 | 0.236 |
| `age` | 0.310 | 0.114 |
| `age_at_diagnosis` | 0.133 | 0.051 |
| `time_since_intake_on` | ~0.0 | 0.177 |
| `time_since_intake_off` | ~0.0 | 0.092 |
| `cohort` | n/a (catégoriel) | 0.090 |
| `gene` | n/a (catégoriel) | 0.052 |
| `sexM`, `rater_id`, `patient_id` | proche de 0 | < 0.03 |

**⚠ Alerte de fuite de données (Leakage) — `off` et `on` :** `off` a un r de Pearson = 0.871 avec `target` ; `on` a un r = 0.669. Les deux sont des sous-scores UPDRS mesurés lors de la même visite que la cible. Si la tâche consiste à prédire le score UPDRS total *à partir de sous-scores enregistrés lors de la même visite*, ces variables sont valides mais sont en réalité des mesures quasi-directes de la cible. Si la tâche consiste à *prédire un score futur à partir de variables passées*, ces colonnes pourraient constituer une **fuite de données** (les mesures futures n'étant pas disponibles au moment de la prédiction). **Ceci doit être confirmé avec la définition de la tâche.**

**Variable ↔ Variable :**
- `age_at_diagnosis` ↔ `age` : r de Pearson = 0.942 — relation quasi-linéaire (l'âge n'est que l'âge au diagnostic + la durée de la maladie). L'une d'elles peut être redondante, ou bien leur différence (la durée de la maladie) peut s'avérer plus informative.
- `on` ↔ `off` : r de Pearson = 0.872 — les scores UPDRS ON et OFF sont fortement corrélés.

## Implications pour la modélisation

- **Tâche de régression** — utilisez la RMSE / MAE comme métriques principales. Pas besoin de stratification. Le `KFold` standard est le comportement par défaut ; envisagez un `GroupKFold` sur `patient_id` (voir ci-dessous).
- **`patient_id` se répète sur plusieurs lignes** → un `KFold` standard fera "fuiter" le même patient à la fois dans les plis d'entraînement et de validation. Un **`GroupKFold` sur `patient_id`** (ou `LeaveOneGroupOut`) est la méthode de séparation correcte pour obtenir une estimation honnête de la généralisation sur de nouveaux patients.
- **Forte proportion de valeurs manquantes dans `off`, `on`, `ledd`, `time_since_intake_on/off`** — le pipeline doit gérer les NaN (imputation, ou estimateurs basés sur les arbres qui les supportent nativement).
- **Risque de fuite (leakage) avec `off` / `on`** — avant l'entraînement, confirmez avec le responsable de la tâche si ces colonnes représentent des mesures simultanées ou un historique décalé (lagged) valide. Si elles sont simultanées, envisagez de faire un entraînement avec et sans celles-ci pour quantifier leur effet.
- **Colinéarité entre `age_at_diagnosis` et `age`** — envisagez d'utiliser la durée de la maladie = `age − age_at_diagnosis` comme variable construite (feature engineering) à la place de l'une d'elles.
- **Effet `rater_id`** — 50 évaluateurs ; un encodage catégoriel (target encoding ou l'encodeur de catégories "sales" de `skrub`) pourrait capturer le biais au niveau de l'évaluateur.
- **Queue asymétrique vers la droite (scores > 77)** — ~200 lignes sur 44 590. Une transformation de la cible par logarithme ou racine carrée pourrait améliorer l'homoscédasticité des résidus, mais la cible est par ailleurs presque normale ; à évaluer de manière empirique.
- **Aucune structure temporelle détectée** — un `TimeSeriesSplit` n'est pas indiqué, à moins qu'une colonne de date de visite ne soit découverte.

## Questions ouvertes

1. **Fuite (leakage) de `off` et `on`** — les sous-scores UPDRS `on`/`off` sont-ils mesurés lors de la même visite que la `target` ? Si oui, leur corrélation r ≈ 0.87/0.67 avec la `target` encode très probablement une information quasi-directe sur laquelle un modèle s'appuiera de façon triviale. Clarifiez le scénario de prédiction.
2. **Définition de la `target`** — la `target` est-elle la même chose ou est-elle dérivée de `on` + `off` ? Si la `target` est une combinaison linéaire de `on` et `off`, le r de Pearson est mécaniquement logique.
3. **Signification de `time_since_intake_on/off`** — corrélation de Pearson de ~0 mais V de Cramér modéré (0.18 / 0.09) ; le schéma des valeurs manquantes (46 % et 79 %) suggère que la plupart des visites ne l'ont pas enregistré. Les données manquantes sont-elles aléatoires ou structurées (par ex. seules les visites en état OFF ont un `time_since_intake_off`) ?
4. **Deux cohortes (A / B)** — diffèrent-elles par leur protocole de collecte, leur population de patients ou leur période temporelle ? Un indicateur de cohorte devra peut-être être inclus comme variable ou utilisé pour une séparation stratifiée.
5. **Valeurs manquantes pour `gene`** — 32 % de manquants. Est-ce manquant de façon aléatoire, ou certains sous-types génétiques ne sont-ils systématiquement pas testés ?
