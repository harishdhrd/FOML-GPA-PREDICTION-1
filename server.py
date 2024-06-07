from flask import Flask, url_for, render_template, redirect, request,jsonify
from get_details import GetDetails
import time
import pandas as pd

app = Flask(__name__)


details = GetDetails()
details.map_id_to_reg_no()

@app.route("/")
def home():
    return render_template("index.html",reg_no_list = sorted(details.get_reg_no()))

@app.route("/internals",methods=["GET","POST"])
def internals_page():
    if request.method == "POST":
        reg_no = request.form['reg_no']
        internal_marks = details.getInternalMarks(reg_no)
        # actual_gpa = details.find_gpa(details.get_mapped_id(reg_no))
        actual_gpa = details.find_gpa_from_csv(int(reg_no))
        data = [reg_no,details.get_mapped_dict(),internal_marks,actual_gpa]
        return render_template('internals.html',data = data)
    return redirect(url_for('home'))


@app.route("/predict",methods=["GET","POST"])
def prediction_page():
    if request.method == "POST":
        reg_no = request.form['reg_no']
        # actual_gpa = details.find_gpa(details.get_mapped_id(reg_no))
        actual_gpa = details.find_gpa_from_csv(int(reg_no))
        model = details.train_model("student-data.csv")
        internal_marks = details.get_user_internals_from_csv(int(reg_no))
        predicted_val = model.predict(internal_marks)[0][0]
        data = [reg_no, round(actual_gpa,2), round(predicted_val,2)]
        return render_template('prediction.html',data=data)


# @app.route("/add-details")
# def haha():
#     return jsonify({"Response":details.add_details()})

if __name__ == "__main__":
    app.run(debug=True,port=8081)