import requests
import json
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



class GetDetails(object):
    def __init__(self):
        self.mapped_reg_no = {

        }
        self.reg_no = []

    def map_id_to_reg_no(self):
        url = "http://rajalakshmi.in/UI/Modules/UniCurrentResult.aspx/GetStudent"

        #cse 2020 ->{"Batch":"4032","CourseId":"11","SemesterId":"3088"}

        json_data = {"Batch":"4032","CourseId":"11","SemesterId":"3088"}

        response = json.loads(requests.post(url,json=json_data).json()['d'])
        # return response
        for res in response:
            reg_no = res['Name'].split("-")[1].strip()
            personId = res['PersonId']
            self.mapped_reg_no[reg_no] = personId
            self.reg_no.append(reg_no)
    


    def find_gpa(self,PersonID):
        cookie = "Cookie: G_AUTHUSER_H=1; G_ENABLED_IDPS=google; ASP.NET_SessionId=uuw54k205qz1hunfs1vbvrql"
        json_data = {"AcademicYearId":"3382","CourseId":11,"SemesterId":"3088","PersonId":PersonID}
        headers = {"Referer": "http://rajalakshmi.in/UI/Modules/uniCurrentResult.aspx?FormHeading=Examination%20Result","Cookie":"G_ENABLED_IDPS=google; ASP.NET_SessionId=ixsjiaxljkc1u4ab1zttydmd"}
        response = json.loads(requests.post("http://rajalakshmi.in//UI/Modules/UniCurrentResult.aspx/GetResult",json=json_data,headers=headers).json()['d'])
        gpa,sub_count = 0,0
        for res in response:
            if res['TOTAL'] != 0 and res['CREDITS'] != 0:
                gpa += res['TOTAL'] / res['CREDITS']
                sub_count += 1
        if sub_count != 0:
            return gpa/sub_count
        return gpa/1


    def find_gpa_from_csv(self,reg_no):
        data = pd.read_csv("student-data.csv")
        no = data.loc[data["Reg no"] == reg_no,["GPA"]].values
        print(no)
        return no[0][0]

    def add_details(self):
        with open("student-data.csv","a",newline='\n') as file:
            writer = csv.writer(file)
            subs = ["Computer Graphics","Computer Networks","Internet Programming","Introduction to Robotic Process Automation","Principles of Artificial Intelligence","Supply Chain Management","Theory of Computation","GPA"]
            writer.writerow(["Reg no"] + subs)
            for reg_no in sorted(self.reg_no):
                if reg_no == str(200701510):
                    continue
                sep_list = self.getInternalMarks(reg_no)
                t1 = [reg_no]
                
                
                for sub in subs:
                    tot = 0
                    found = 0
                    t2 = [0]
                    for haha in sep_list['CAT 1']:
                            #print(haha)
                        for hehe in haha:
                                #print(hehe)
                            if hehe == sub:
                                tot += haha[hehe] if haha[hehe] is not None else 0
                                found = 1
                                break
                        if found:
                            t2.append(tot)
                            break
                    if found != 1:
                        t2.append(0)


                    found = 0
                    tot = 0
                    for haha in sep_list['CAT 2']:
                            #print(haha)
                        for hehe in haha:
                                #print(hehe)
                            if hehe == sub:
                                tot += haha[hehe] if haha[hehe] is not None else 0
                                found = 1
                                break
                        if found:
                            t2.append(tot)
                            break
                    if found != 1:
                        t2.append(0)


                    found = 0
                    tot = 0
                    for haha in sep_list['CAT 3']:
                            #print(haha)
                        for hehe in haha:
                                #print(hehe)
                            if hehe == sub:
                                tot += haha[hehe] if haha[hehe] is not None else 0
                                found = 1
                                break
                        if found:
                            t2.append(tot)
                            break
                    if found != 1:
                        t2.append(0)
                    

                    t1.append(sum(t2.copy()))

                #print(t1)
                PersonID = self.get_mapped_id(reg_no)
                gpa = self.find_gpa(PersonID)
                t1.append(gpa)
                writer.writerow(t1)
            return "[+]Successfully Added"




    def getInternalMarks(self,reg_no):
        PersonID = self.get_mapped_id(reg_no)
        params = {
            "PersonId":int(PersonID),
            "Semester":5,
            "Category":0
	    }
        match = {
        "1":"I",
        "2":"II",
        "3":"III",
        "I":"I",
        "II":"II",
        "III":"III"
        }
        sep_list = {}

        s = requests.Session()

        marks_list = json.loads(s.post("http://rajalakshmi.in/UI/Modules/HRMS/ManageStaffStudent/UniPersonInfo.asmx/BindInternalMarks", json = params).json()['d'])
        for mark in marks_list:
            marks = mark['EventTitle'].split("/")[2]
            subjName = mark['SubjName']
            tot = mark['Total']
            title = None
            if "ASS" in marks:
                title = marks
                l = sep_list.get(title,[])
            elif '2' in marks:
                l = sep_list.get("CAT 2",[])
                title = "CAT 2"
            elif '1' in marks:
                l = sep_list.get("CAT 1",[])
                title = "CAT 1"
            elif '3' in marks:
                l = sep_list.get("CAT 3",[])
                title = "CAT 3"
            else:
                l = sep_list.get("ASS",[])
                title = "ASS"
            l.append({
                subjName:tot
                })

            sep_list[title] = l
        return sep_list



    def train_model(self,file_name):
        data = pd.read_csv(file_name)
        data = data.drop(["Empty"],axis=1)
        x = data.drop(["GPA"], axis=1)
        y = data[["GPA"]]
        poly=PolynomialFeatures(degree=2)
        x_poly=poly.fit_transform(x)
        model1=LinearRegression.fit(x_poly,y.values)


        model = LinearRegression().fit(x.values,y.values)
        print(model+" "+model1)
        return model

    def get_user_internals_from_csv(self,reg_no):
        data = pd.read_csv('student-data.csv')
        data = data.drop(["Empty"],axis=1)
        x = data.drop(["GPA"], axis=1)
        return x.loc[x['Reg no'] == reg_no,:].values
        


    def get_mapped_id(self, reg_no):
        return self.mapped_reg_no[reg_no]

    def get_mapped_dict(self):
        return self.mapped_reg_no

    def get_reg_no(self):
        return self.reg_no