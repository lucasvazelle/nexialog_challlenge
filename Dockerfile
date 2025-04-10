# Utilise une image Python légère
FROM python:3.10-slim

# Empêche la création de fichiers .pyc et affiche les logs en temps réel
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dossier de travail dans le container
WORKDIR /app

# Installe les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie les dépendances Python et les installe
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le code dans le container
COPY . .

# Expose le port 8080 (nécessaire pour GCP)
EXPOSE 8080

# Lance l'app Dash avec gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:server"]
