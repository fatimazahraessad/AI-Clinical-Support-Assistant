from flask import Flask, render_template, request
from orchestrator import orchestrate
import os

# ---------- INITIALISATION FLASK ----------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------- ROUTE PRINCIPALE ----------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        # Récupération du texte
        text_input = request.form.get("text_input")
        print("Texte reçu :", text_input)

        # Récupération du fichier
        uploaded_file = request.files.get("file")
        file_path = None

        if uploaded_file and uploaded_file.filename != "":
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
            uploaded_file.save(file_path)
            print("Fichier sauvegardé à :", file_path)
        else:
            print("Aucun fichier uploadé")

        # Appel de la fonction orchestrate
        result = orchestrate(text=text_input, file_path=file_path)
        print("Résultat orchestrate :", result)

    # Rendu du template avec le résultat
    return render_template("index.html", result=result)

# ---------- LANCEMENT DU SERVEUR ----------
if __name__ == "__main__":
    app.run(debug=True, port=5000)