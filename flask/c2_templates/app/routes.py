from flask import render_template
from app import app

@app.route('/') # decorator
@app.route('/index') # decorator
def index():
    user = {'username': 'whoji'}
    posts = [
        {
            'author': {'username': 'John'},
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': {'username': 'Susan'},
            'body': 'The Avengers movie was so cool!'
        }
    ]
    #return "Hello, World!"
    return render_template('index.html', title='Home', user=user
        ,posts=posts)

@app.route('/test1') # decorator
def test1():
    return "Hello, Test 1 !"