## Awaaz Monorepo

This repository contains two separate projects:

- `Awaaz-Backend`: Django + AI backend
- `Awaaz-App`: React Native (Expo) mobile app

### Backend (Awaaz-Backend)
- Tech: Django 5, DRF, custom AI services
- Location: `Awaaz-Backend/`
- Run (from `Awaaz-Backend/`):
```
source venv/bin/activate  # if present
python manage.py runserver 0.0.0.0:8000
```

### Frontend (Awaaz-App)
- Tech: React Native (Expo)
- Location: `Awaaz-App/`
- Run (from `Awaaz-App/`):
```
npm install
npx expo start
```

### Communication
- The frontend communicates with the backend exclusively via HTTP API calls (e.g., `fetch` or `axios`).
- Configure the backend base URL in `Awaaz-App/app.json` under `expo.extra.apiBaseUrl` (use your machine IP for physical devices).

### EAS Builds (Android/iOS)
- Only work inside `Awaaz-App/` for EAS builds.
- Example (APK preview build):
```
cd Awaaz-App
npx eas build -p android --profile preview
```

### Notes
- Keep backend and app concerns isolated. Do not place backend files in `Awaaz-App/`, and do not place mobile assets in `Awaaz-Backend/`.
## Pothole Severity Classifier (Local, no API)

- Classes: `minor`, `moderate`, `severe`
- Put images into `data/potholes/train|val|test/<class>/...jpg`
- Create and activate venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install deps: see below.

### Install
```
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install streamlit opencv-python pillow numpy scikit-learn rich pyyaml
```

### Train
```
python src/train/train.py --data_dir data/potholes --epochs 15 --batch_size 32
```

### Inference
```
python src/app/predict.py --checkpoint checkpoints/best.pt --image /path/to/image.jpg
```

### Streamlit app
```
streamlit run src/app/app.py
```
