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

Parsing du jeu de données:

```bash
python extract.py
```
train.txt -> train_proper.csv