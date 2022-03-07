from flask import Flask, render_template, request
import model

app = Flask(__name__, template_folder='./')

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method.upper() == 'GET':
        return render_template('index.html')
    else:
        #get username from submitted form
        username = request.form['txtUserID']
        if username == "":
            return render_template('index.html', error="Missing username in input.", result=[])
        else:
            result = model.get_recommendation(username)
            return render_template('index.html', results=result)


if __name__ == '__main__':
    app.run()