# Fitness Activity Classification

An end-to-end MLOps starter project. It trains a Random Forest classifier that predicts a person's activity (**resting, walking, running, cycling**) from wearable sensor readings, serves it through a Flask app with a web form and JSON API, tests it with pytest, and packages it with Docker.

## Project structure

```
mlops project
├── app.py              # Flask app (web UI + /predict API)
├── train_model.py      # Generates data, trains the model, saves model.pkl
├── model.pkl           # Trained model artifact
├── requirements.txt
├── .gitignore
├── Dockerfile
├── README.md
├── templates/
│   └── index.html      # Web form
├── static/
│   └── style.css
└── tests/
    └── test_app.py
```

## Features used

| Feature | Meaning |
|---|---|
| `heart_rate` | Heart rate in beats per minute |
| `accel_mean` | Average acceleration magnitude (g) |
| `accel_std` | Variation of acceleration (g) |
| `cadence` | Steps per minute (walking/running) or pedal rpm (cycling) |

The training data is synthetic so the project works out of the box. To use real data, replace `generate_data()` in `train_model.py` with a loader for your own CSV that has these columns plus an `activity` label.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train_model.py            # creates model.pkl
python app.py                    # http://localhost:5000
```

## API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"heart_rate": 155, "accel_mean": 1.8, "accel_std": 0.9, "cadence": 170}'
```

Response:

```json
{"activity": "running", "probabilities": {"cycling": 0.0, "resting": 0.0, "running": 1.0, "walking": 0.0}}
```

`GET /health` returns `{"status": "ok"}` for health checks.

## Tests

```bash
pytest
```

## Docker

```bash
docker build -t fitness-activity-classifier .
docker run -p 5000:5000 fitness-activity-classifier
```

## Next steps

- Add CI (GitHub Actions) that runs `train_model.py` and `pytest` on every push.
- Track experiments and model versions with MLflow.
- Add monitoring for prediction drift.
