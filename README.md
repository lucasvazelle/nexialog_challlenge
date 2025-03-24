# 📡 Nexialog Challenge 2025 – Détection d’anomalies réseau

Bienvenue dans notre projet de data science réalisé dans le cadre du **Challenge Nexialog x Master MoSEF**, en partenariat avec **SFR**.  
Notre objectif est de construire un modèle de **détection d’anomalies sur le réseau SFR**, à partir des données issues de tests DNS et de scoring réalisés sur les box des clients.

---

## Objectif du projet

Identifier les zones du réseau (OLT, PEAG...) susceptibles de subir des **interruptions simultanées**, **sans supervision** (pas de variable cible), en exploitant les résultats des tests techniques effectués sur les box client.

---

## Structure du projet

#### 📁 `S1: analyse_exploratoire_V1.ipynb`
- Preprocessing des données brutes: Nettoyage, vérifications, premières statistiques descriptives.
- Identification des colonnes pertinentes (tests DNS / tests scoring).

#### 📁 `S2: etape2_finale.ipynb`
- Agrégation des données par `date-hour` et `olt_name`.
- Data Engineering : rajout de variables potentiellement pertinentes pour la détéction d'anomalies
  
#### 📁 `S3: séparation des tables et data viz`
- Séparation des jeux de données (à ne pas faire) .
- Data visualisation exploratoire sur différentes combinaisons (OLT, PEAG, etc.).

#### 📁 `S4: etape4_model_sans_ml.ipynb`
- Modélisation sans un modèle de ML.
- Utilisation de **méthodes de "jump detection"** (changements brusques) pour identifier des ruptures dans les indicateurs de performance (latence, etc.).
  
---
