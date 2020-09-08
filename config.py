#Img2json
image_dir = 'img/left/0507_left'
json_dir = ''




#json2h5
#0_rename_img
img_folder = "../Img2json/img/ai_add_searchedyoga_test"
dest_folder = "../Img2json/img/ai_add_searchedyoga_test_renamed"

#1_renamejson
#ceiling pose:3
#underwater pose:2
#yoga+ai: 0
imageandjson_first_number = "3"

number_json_file = 2
json_filename = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/json2h5/json/0612_underwater_all.json'
output_jsonname = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/json2h5/json/0612_underwater_all_rename.json'

#2_json2h5_others/2_json2h5_kp2json
input_json = ["json/hm_wrong.json"]
output_h5name = 'h5/hm_wrong.json.h5'
