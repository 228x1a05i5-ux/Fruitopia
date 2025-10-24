# Fruitopia - Fruit Classification & Recipe Suggestion (MVP)

This is a minimal MVP of the Fruitopia project you described. It provides a Flask web app that accepts an uploaded image, uses a simple heuristic classifier (based on dominant color) to guess the fruit, and returns a suggested recipe.

## What is included
- `flask/` - contains the Flask app, templates and static files
    - `app.py` - main Flask backend (uses a color heuristic classifier)
    - `templates/` - `index.html`, `predict.html`
    - `static/css/style.css` - simple styling
- `uploads/` - folder where uploaded images are saved
- `models/` - placeholder for trained models (not included)
- `Fruit-Images-Dataset-master/` - placeholder dataset folder
- `requirements.txt` - Python packages to install
- `README.md` - this file

## How to run locally (recommended via Anaconda)
1. Create a virtual environment (optional but recommended):
   ```bash
   conda create -n fruitopia python=3.9
   conda activate fruitopia
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   cd flask
   python app.py
   ```
4. Open your browser at http://127.0.0.1:5000/

**Deno Video Link**


## Next steps to integrate a trained CNN model
- Train a Keras/TensorFlow model in `Fruit Classification.ipynb` or your own notebook.
- Save the model as `models/fruit.h5` or outside and update `app.py` to load and use the Keras model.
- Replace the `heuristic_fruit_classifier` with a keras model inference function.

## Notes
- This MVP uses a heuristic classifier so it will not be as accurate as a trained CNN. It's intended to be a working scaffold you can replace with the trained model later.
- If you want, I can include an example Keras model file (very small) or code to train on a subset of images. Ask me and I'll add it to the ZIP.
