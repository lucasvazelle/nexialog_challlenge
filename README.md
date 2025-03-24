# 📡 Nexialog Challenge 2025 – Détection d’anomalies réseau

Bienvenue dans notre projet de data science réalisé dans le cadre du **Challenge Nexialog x Master MoSEF**, en partenariat avec **SFR**.  
Notre objectif est de construire un modèle de **détection d’anomalies sur le réseau SFR**, à partir des données issues de tests DNS et de scoring réalisés sur les box des clients.

---

## Objectif du projet

Identifier les zones du réseau (OLT, PEAG...) susceptibles de subir des **interruptions simultanées**, **sans supervision** (pas de variable cible), en exploitant les résultats des tests techniques effectués sur les box client.

---

## Structure du projet

#### 📁 `S1: analyse_exploratoire_V1.ipynb`
- **Étape 1** : Preprocessing des données brutes.
- Nettoyage, vérifications, premières statistiques descriptives.
- Identification des colonnes pertinentes (tests DNS / tests scoring).

#### 📁 `S2: etape2_finale.ipynb`
- **Étape 2** : Agrégation des données par `date-hour` et `olt_name`.
- Mise en forme de la base pour préparer l’analyse temporelle à la maille des nœuds réseau.

#### 📁 `S3: séparation_des_tables_data_viz.ipynb`
- **Étape 3** : (à priori à ne pas faire) Séparation des jeux de données par test.
- Data visualisation exploratoire sur différentes combinaisons (OLT, PEAG, etc.).

#### 📁 `S4: etape4_model_sans_ml.ipynb`
- **Étape 4** : Modélisation sans machine learning.
- Utilisation de **méthodes de "jump detection"** (changements brusques) pour identifier des ruptures dans les indicateurs de performance (latence, retransmission, etc.).
- Premiers résultats et anomalies détectées à certains niveaux du réseau.

---
