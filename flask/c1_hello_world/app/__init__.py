from flask import Flask

app = Flask(__name__)

from app import routes


'''
The routes are the different URLs that the application implements.
In Flask, handlers for the application routes are written as Python
functions, called view functions. View functions are mapped to one
 or more route URLs so that Flask knows what logic to execute when
 a client requests a given URL.
'''