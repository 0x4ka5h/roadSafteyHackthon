from crypt import methods
from flask import Blueprint, jsonify, request
from flask_login import login_required
from ..model import vehicleDetails
from sqlalchemy import text,select
from .. import db
ownerGPSauth = Blueprint("ownerGPSauth",__name__)

######################    current gps location      ####################################

@ownerGPSauth.route("/api/currentStatus/",methods=['POST'])

def currentStatus():
    longitude = str(request.json.get('lon'))
    latittude = str(request.json.get('lat'))

    
    result=db.session.execute(text("SELECT gpsPoint FROM vehicle_details"))
    result = result.mappings().all()


    
    try:
        return jsonify({"gpsLocation":str(result[-1]['gpsPointCurr_'])}),201   #last gps point that vehicle left
    except:
        return "No Data found",200

