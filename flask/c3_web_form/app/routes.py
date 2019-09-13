from flask import render_template, flash, redirect, url_for
from app import app
from app.forms import LoginForm

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

@app.route('/login', methods=['GET','POST']) # decorator
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for user {}, remember_me={}'.format(
            form.username.data, form.remember_me.data))
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign in', form=form)
