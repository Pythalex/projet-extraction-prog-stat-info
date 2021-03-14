# Projet M2 extraction et programmation statistique de l'information

ELANBARI Anass
BONIN Alexandre

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

#### Modèle BERT :
Se trouve sur le notebook : BERT_Model.ipynb
Ce notebook fait du fine-tuning sur les tweets d'un modèle BERT pré-entrainé et sauvegarde sous tf_models/BERT
Ce modèle a été utilisé pour prédire les labels du corpus test
