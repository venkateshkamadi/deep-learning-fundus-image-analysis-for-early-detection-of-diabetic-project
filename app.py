import os, json
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ---------------- APP SETUP ----------------
app = Flask(__name__)
app.secret_key = "super_secure_dr_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- USER STORAGE ----------------
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

USERS = load_users()

# ---------------- LOGIN REQUIRED ----------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ---------------- MODEL ----------------
model_path = os.path.join(BASE_DIR, "model", "Updated-Xception-diabetic-retinopathy.h5")
model = load_model(model_path)

CLASSES = [
    "No Diabetic Retinopathy",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

LEVEL_INFO = {
    "No Diabetic Retinopathy": "Healthy retina. No treatment needed.",
    "Mild": "Early stage. Regular monitoring advised.",
    "Moderate": "Blood vessel damage present. Doctor consultation required.",
    "Severe": "High risk of vision loss. Immediate treatment needed.",
    "Proliferative DR": "Critical stage. Surgery or laser treatment required."
}

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if email in USERS:
            flash("User already exists", "danger")
            return redirect(url_for("register"))

        USERS[email] = generate_password_hash(password)
        save_users(USERS)
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    # If already logged in, skip login page
    if "user" in session:
        return redirect(url_for("predict"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Empty input check
        if not email or not password:
            flash("Please enter email and password", "danger")
            return redirect(url_for("login"))

        # Credential check
        if email in USERS and check_password_hash(USERS[email], password):
            session["user"] = email
            flash("Login successful!", "success")
            return redirect(url_for("predict"))
        else:
            flash("Wrong email or password", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "success")  # ✅ MUST be success
    return render_template("logout.html")





@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    prediction = confidence = image_url = explanation = None

    if request.method == "POST":
        file = request.files["image"]
        if file.filename == "":
            flash("Please upload an image", "danger")
            return redirect(request.url)

        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        img = image.load_img(path, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0

        preds = model.predict(img)[0]
        idx = np.argmax(preds)

        prediction = CLASSES[idx]
        confidence = round(float(preds[idx]) * 100, 2)
        explanation = LEVEL_INFO[prediction]
        image_url = f"uploads/{file.filename}"

    return render_template(
        "prediction.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        explanation=explanation
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
