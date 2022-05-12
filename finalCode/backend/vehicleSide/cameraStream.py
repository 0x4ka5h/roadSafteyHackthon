import base64
import cv2,numpy as np
from sqlalchemy import text
from flask import Blueprint, jsonify, redirect, request, url_for
from .. import db

cameraStreamauth = Blueprint("cameraStreamauth",__name__)

@cameraStreamauth.route("/api/stream/estimateDepth/",methods=['POST'])
def estimateDepth():

    imageData = request.json.get('imageData')
    
    ##testing image
    
    if int(camIndex_) == 1:
        dec = base64.b64decode(imageData)
        nparr = np.fromstring(dec, np.uint8)
        image_ = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite("apiCenter/static/1/view.png",image_)
        return jsonify({"request":"successfull"}),201

    elif int(camIndex_) == 2:
        dec = base64.b64decode(imageData)
        nparr = np.fromstring(dec, np.uint8)
        image_ = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite("apiCenter/static/2/view.png",image_)
        return jsonify({"request":"successfull"}),201
