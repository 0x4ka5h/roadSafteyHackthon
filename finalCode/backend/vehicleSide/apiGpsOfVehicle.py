from flask import Blueprint, jsonify, redirect, request, url_for
from ..model import vehicleDetails
from .. import db

vehicleGPSauth = Blueprint("vehicleGPSauth",__name__)


@vehicleGPSauth.route("/api/vehicle/sendDetails/" , methods = ['POST'])

def sendDetails():
    
    gpsPoint = request.json.get('gpsPoint')

    kind = request.json.get('kind')
    depth = request.json.get('depth')

    details_ = vehicleDetails(gpsPoint=gpsPoint, size_ = kind, depth_ = depth)
    db.session.add(details_)
    db.session.commit()   # add the details to the database

    return jsonify({'request': "Success"}), 201
