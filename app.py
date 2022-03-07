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
            dict_result = model.get_recommendation(username)
            error = "Sorry! An error occured while serving you." if dict_result['error'] != "" else ""
            result = dict_result['result']
            return render_template('index.html', error=error, results=result)


if __name__ == '__main__':
    app.run()