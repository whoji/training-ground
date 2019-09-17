import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    '''
    it is in general a good practice to set configuration from
    environment variables, and provide a fallback value when the
    environment does not define the variable.
    '''
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'

    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False