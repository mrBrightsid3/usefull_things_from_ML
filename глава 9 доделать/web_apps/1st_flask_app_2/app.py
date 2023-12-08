from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

app = Flask(__name__)


class HelloForm(Form):
    sayhallo = TextAreaField("", [validators.DataRequired()])


@app.route("/")
def index():
    form = HelloForm(request.form)
    return render_template("first_app.html", form=form)


@app.route("/hello")
def sayhello():  # визуализирует html страницу hello
    form = HelloForm(request.form)
    if request.method == "POST" and form.validate():
        name = request.form["sayhallo"]
        return render_template("hello.html", name=name)
    return render_template("first_app.html", form=form)


if __name__ == "__main__":
    app.run(debug=True)
