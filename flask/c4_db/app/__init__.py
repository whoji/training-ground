from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)
migrate = Migrate(app, db) # migration engine. wtf is this ?

from app import routes
from app import models # This module will define the structure of the database.
'''
The routes are the different URLs that the application implements.
In Flask, handlers for the application routes are written as Python
functions, called view functions. View functions are mapped to one
 or more route URLs so that Flask knows what logic to execute when
 a client requests a given URL.
'''
