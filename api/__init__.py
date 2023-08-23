from flask import Flask

def create_app():
    app = Flask(__name__)

    @app.route("/about")
    def about():
        return "about"
    
    @app.route("/home")
    def home():
        return "home"
    
    return app