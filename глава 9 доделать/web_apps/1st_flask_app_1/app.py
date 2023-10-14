from flask import Flask, render_template

app = Flask(
    __name__
)  # теперь фреймворку фласк известно что подкаталог HTML шаблонов располагается в том же каталоге


@app.route(
    "/"
)  # декоратор маршрута чтобы указать URL который должен инициализировать выполнение функции index
def index():
    return render_template("first_app.html")


if __name__ == "__main__":
    app.run()
