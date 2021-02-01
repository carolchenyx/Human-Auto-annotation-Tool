#Img2json

image_dir = 'img/yoga_3'
json_dir = 'img/yoga_3'
yolo_cfg = "yolo/cfg/yolov3-spp.cfg"
yolo_weight = 'models/yolo/yolov3-spp.weights'
model_path = '../../PoseTrainingPytorch/exp/ceiling_tmp/1/1_best.pkl'



#json2h5
#0_rename_img
img_folder = "/home/hkuit155/Documents/yo
dest_folder = "/home/hkuit155/Documents/yo"

#1_renamejson
#ceiling pose:3
#underwater pose:2
#yoga+ai: 0
imageandjson_first_number = "1"

number_json_file = 2
json_filename = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/json2h5/json/r_all.json'
output_jsonname = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/json2h5/json/_all_rename.json'

#2_json2h5_others/2_json2h5_kp2json
input_json = ["/home/hkuit155/Documents/yo_train.json"]
output_h5name = '/home/hkuit155/Documents/yo_train.h5'
