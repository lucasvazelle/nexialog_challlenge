# 📡 Nexialog Challenge 2025 – Détection d’anomalies réseau

Bienvenue dans notre projet de data science réalisé dans le cadre du **Challenge Nexialog x Master MoSEF**, en partenariat avec **SFR**.  
Notre objectif est de construire un modèle de **détection d’anomalies sur le réseau SFR**, à partir des données issues de tests DNS et de scoring réalisés sur les box des clients.

---

## Lancer l'application de détection d'anomalies

### En local

1. Installer les dépendances
$ pip install -r requirements.txt

2. Lancer
$ python app.py

### Avec Docker

1. Installer docker 
$ sudo apt docker

2. Build l'image OU pull l'image
$ docker build -t nexialog .
ou
$ docker pull lucasvazelle/nexialog

3. Lancer
$docker run nexialog
$docker run lucasvazelle/nexialog

    ### Sur le web

https://nexialogchallenge-212301858764.us-central1.run.app

## Structure du projet

#### `src.domain.Pre_traitement_data*' et 'src.sandbox.*'
- Preprocessing des données brutes: Nettoyage, vérifications, premières statistiques descriptives.

####`src.domain.Agregation_at_level_OLT_PEAG.ipynb'
- Agrégation des données
- Feature  Engineering
-
#### `src.models;*.ipynb`
- Méthodes économétrique :  **jump**
- Méthode **DBSCAN**
- Méthode **isolation forest**

#### `src.app.`
- Application dash permettant la visualisation des résultats de nos modèles de détection

