#Img2json
image_dir = 'img/ceiling_pre_bad/image'
json_dir = ''




#json2h5
#0_rename_img
img_folder = ""
dest_folder = "img/0612_underwater"

#1_renamejson
#ceiling pose:0,2,4,6,8,....
#underwater pose:1,3,5,7,9,...
imageandjson_first_number = "3"

number_json_file = 2
json_filename = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/json2h5/json/0612_underwater_all.json'
output_jsonname = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/json2h5/json/0612_underwater_all_rename.json'

#2_json2h5_others/2_json2h5_kp2json
input_json = ['json/0612_underwater_all_rename.json','json/0612_underwater_all_rename.json']
output_h5name = 'h5/0612_underwater_all_rename.h5'