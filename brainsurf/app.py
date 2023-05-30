from flask import Flask, render_template
import subprocess

app = Flask(__name__, template_folder="templates")


@app.route("/")
def home():
    return render_template("landing.html")  # Use your own HTML design for the home page


# @app.route("/documentation")
# def documentation():
#     return render_template("documentation.html")  # Use your own HTML design for the documentation page


@app.route("/ide")
def ide():
    subprocess.Popen(["streamlit", "run", "streamlit_app.py"])
    return render_template("templates\streaming_html.html")


if __name__ == "__main__":
    app.run(debug=True)

