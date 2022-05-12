from flask_login import UserMixin
from sqlalchemy import PrimaryKeyConstraint
from . import db


class vehicleDetails(UserMixin,db.Model):
    id=db.Column(db.Integer, primary_key=True)
    gpsPoint = db.Column(db.String(100))
    size_ = db.Column(db.String(100))
    depth_ = db.Column(db.String(100))