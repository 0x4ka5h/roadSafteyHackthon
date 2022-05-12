from flask import Blueprint, jsonify
from . import db
from flask_login import current_user

main = Blueprint('main', __name__)


@main.route('/')
def home():
    return jsonify({"status":"testing","message":" have fun in anything..!"})

