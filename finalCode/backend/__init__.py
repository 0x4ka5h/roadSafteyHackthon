from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# init SQLAlchemy so we can use it later in our models
db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'g00g1y5p4'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

    db.init_app(app)

    # home route and vehicleStatus route
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    ##########################    ownerSide        ###############################

    from roadSafteyHackthon.ownerSide.apiGpsToOwner import ownerGPSauth
    app.register_blueprint(ownerGPSauth)



    ##########################    vehicleSide      ###############################
    from roadSafteyHackthon.vehicleSide.apiGpsOfVehicle import vehicleGPSauth
    app.register_blueprint(vehicleGPSauth)

    from roadSafteyHackthon.vehicleSide.cameraStream import cameraStreamauth
    app.register_blueprint(cameraStreamauth)
    return app
