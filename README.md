
# Capability Assessment Web App

Streamlit-App für Personalbewertung mit Soll-Profilen aus einer Excel-"Capability Matrix" mit **zweizeiligem Header**:
- **Zeile 1 (Spalten F–AM):** Team-Namen
- **Zeile 2 (Spalten F–AM):** Positions-Namen
- **Zeilen darunter:** Skills (links) und Soll-Ausprägungen pro Team/Position (rechts)

## Start lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Dockerfile, .dockerignore, cloudbuild.yaml und ein GitHub-Action-Workflow sind enthalten.
