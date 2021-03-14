# Projet M2 extraction et programmation statistique de l'information

ELANBARI Anass
BONIN Alexandre

Rapport du projet : "Rapport extraction et programmation statistique M2.pdf"

---

## Instructions

Requis: 
- python 3.X
- [virtualenv](https://pypi.org/project/virtualenv/)

### Prérequis

#### Windows

```batch
python -m virtualenv venv

venv\Sources\activate.bat

python -m pip install -r requirements.txt
```
(activate.ps1 en powershell)

#### Linux

```bash
python3 -m virtualenv venv

source venv/bin/activate

python3 -m pip install -r requirements.txt
```

### Preprocessing

Avant de pouvoir exécuter les notebook, il est nécessaire d'effectuer le premier traitement sur le train.txt brut:

```bash
cd src
python extract.py
```

Résultat:
tweet_sent_predictor/data/train.txt -> tweet_sent_predictor/data/train_proper.csv

### Notebooks

Les résultats pour le jeu de test on été produits avec le modèle BERT.

#### Modèle BERT :
Se trouve sur le notebook : BERT_Model.ipynb
Ce notebook fait du fine-tuning sur les tweets d'un modèle BERT pré-entrainé et sauvegarde sous tf_models/BERT
Ce modèle a été utilisé pour prédire les labels du corpus test

#### Baseline
Ce notebook utilise un modèle basique (Bayesien naif) et un traitement simple (Count vector) pour avoir des résultats de base sur lequels juger les modèles.

#### LangDetectionPerformance
Évalue les performances de la détection de la langue par lang_detect

#### SmartPredictor
Utilise le chainage de modèle avec le language detector sur différents modèles. Le lang detector filtre les entrées et ne laisse passer que les tweets anglophones.

#### ChainPredictor
Similaire au SmartPredictor mais ne filtre pas les tweets étrangers pour éviter les erreurs de filtrage, la prédiction est ajoutée aux données d'entrée.

#### XGBoost et LightGBM
Des tests isolés de ces modèles sur les données avec le smartpredictor.



