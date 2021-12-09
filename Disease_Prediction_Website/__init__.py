"""
The flask application package.
"""

from flask import Flask
app = Flask(__name__)

import Disease_Prediction_Website.views
