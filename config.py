#Img2json
image_dir = 'img/tree'
json_dir = 'img/tree.json'




#json2h5
#0_rename_img
img_folder = ""
dest_folder = ""

#1_renamejson
#ceiling pose:0,2,4,6,8,....
#underwater pose:1,3,5,7,9,...
imageandjson_first_number = 3

number_json_file = 2
json_filename = 'json/0612_underwater.json'
output_jsonname = 'json/0612_underwater0.json'

#2_json2h5_others/2_json2h5_kp2json
input_json = ['json/treenew.json','json/treenew.json']
output_h5name = 'h5/treenew.h5'