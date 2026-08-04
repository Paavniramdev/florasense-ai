# FloraSense AI — Backend (Week 2)

FastAPI service wrapping the Week 1 model: species prediction, calibrated
confidence, and Grad-CAM explainability, all in one `/predict` call.

## 1. Get your trained model files here

From your Google Drive (`florasense/export/`), copy these 3 files into a local
`models/` folder next to this README:

```
backend/
├── app/
├── models/                  <-- create this folder
│   ├── florasense_best.pt
│   ├── idx_to_name.json
│   └── calibration.json
├── requirements.txt
└── README.md
```

You can download them from Drive manually, or if running this locally with
Drive synced, just point `FLORASENSE_MODEL_DIR` at the export folder directly
(see step 3).

## 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run the server

```bash
# from the backend/ folder
export FLORASENSE_MODEL_DIR=./models     # Windows (cmd): set FLORASENSE_MODEL_DIR=./models
uvicorn app.main:app --reload --port 8000
```

If you didn't copy the files locally and instead have Google Drive mounted /
synced on this machine, just set `FLORASENSE_MODEL_DIR` to that path instead,
e.g. `export FLORASENSE_MODEL_DIR="/path/to/GoogleDrive/florasense/export"`.

## 4. Test it

Open **http://localhost:8000/docs** — FastAPI's auto-generated Swagger UI lets
you upload an image and hit `/predict` directly from the browser, no extra
tooling needed.

Or via curl:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/flower.jpg"
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Confirms the model loaded, reports class count + calibration temperature |
| `/predict` | POST | Upload an image (`file` field), get back species + confidence + top-5 + Grad-CAM |
| `/docs` | GET | Interactive Swagger UI |

### `/predict` response shape

```json
{
  "predicted_species": "alpine sea holly",
  "confidence": 0.94,
  "is_confident": true,
  "top5": [
    {"species": "alpine sea holly", "confidence": 0.94},
    {"species": "globe thistle", "confidence": 0.02},
    ...
  ],
  "gradcam_image_base64": "iVBORw0KG..."
}
```

`is_confident` is `false` when confidence drops below 0.5 — use this in the
frontend to show an "I'm not sure" state instead of a wrong-but-confident
guess. `gradcam_image_base64` decodes to a PNG showing the original image next
to its Grad-CAM heatmap.

## Notes

- CORS is wide open (`allow_origins=["*"]`) for local dev with the Streamlit
  frontend coming in a later week. Tighten this before any public deployment.
- The model loads once at startup (not per-request) — first request after
  boot may take a second longer while the model is still warming up on GPU
  if applicable.
- This has been tested end-to-end (health check + real image upload + Grad-CAM
  generation) with a dummy checkpoint to confirm the pipeline itself works;
  swap in your real trained weights and it's ready to go.

## Next: Week 3

This backend becomes one of the *tools* the LangGraph agent calls — the agent
decides when to invoke the classifier vs. the RAG retriever vs. the weather
API, based on the user's question.
