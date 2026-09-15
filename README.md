# pandora_dinosaurus

ML / analytics companion to the Pandora **Dino** chicken-farm app.

| Repo | Role |
|------|------|
| **[pandora_public](https://github.com/cvak100/pandora_public)** | Public Dino module — farm ops, SQLite models, heuristic prediction engine. **Source of the farm database.** |
| **This repo (`pandora_dinosaurus`)** | LightGBM weekly egg forecasting, EDA, advanced analysis, Flask dashboard — reads an export of that DB. |

## App

See **[chicken-forecast/README.md](chicken-forecast/README.md)** for setup, data layout, and usage.

```bash
cd chicken-forecast
python -m venv venv
# activate venv, then:
pip install -r requirements.txt
# place farm.db + weather JSON under data/raw/
python app.py
```
