import base64
import requests
import json

map_id_to_reg = {

}

retreive_photo_url = "http://rajalakshmi.in/UI/Modules/HRMS/ManageStaffStudent/UniPersonInfo.asmx/RetrievePersonPhoto"



def removeUnwantedChars(inputString):
	for i in range(len(inputString)):
		cur_info = inputString[i]
		cur_info = cur_info.replace("[","")
		cur_info = cur_info.replace("{","")
		cur_info = cur_info.replace("]","")
		cur_info = cur_info.replace("}","")
		cur_info = cur_info.replace("\"","")
		cur_info = cur_info.replace("\"","")
		cur_info = cur_info.replace("?","")
		inputString[i] = cur_info
	return inputString

def save_photo(url,PersonID,roll_no):
	print("[+]"+"assets/")
	s = requests.Session()
	params = {
		"PersonID":PersonID
	}
	getPhoto = s.post(url,json=params).json()["d"]
	getPhoto = getPhoto.split("*")
	getPhoto = removeUnwantedChars(getPhoto)[0]
	encodedString = getPhoto.split(",")[1]
	file_name = "static\\assets"+"\\"+ roll_no + ".jpg"
	with open(file_name,"wb") as f:
		f.write(base64.b64decode(encodedString))

def get_students(map_id_to_reg):
    url = "http://rajalakshmi.in/UI/Modules/UniCurrentResult.aspx/GetStudent"

    #cse 2020 ->{"Batch":"4032","CourseId":"11","SemesterId":"3088"}

    json_data = {"Batch":"4032","CourseId":"11","SemesterId":"3088"}

    response = json.loads(requests.post(url,json=json_data).json()['d'])
    # return response
    count = 0
    for res in response:
        reg_no = res['Name'].split("-")[1].strip()
        personId = res['PersonId']
        map_id_to_reg[reg_no] = personId
        count += 1
    print(count)
    return map_id_to_reg

map_id_to_reg = get_students(map_id_to_reg)
for reg_no in map_id_to_reg:
	save_photo(retreive_photo_url ,map_id_to_reg[reg_no],reg_no)



