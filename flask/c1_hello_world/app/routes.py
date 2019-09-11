from app import app

@app.route('/') # decorator
@app.route('/index') # decorator
def index():
    return "Hello, World!"

@app.route('/test1') # decorator
def test1():
    return "Hello, Test 1 !"