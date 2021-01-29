#Img2json

# image_dir = 'img/yoga_3'
# json_dir = 'img/yoga_3'
image_dir = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/Img2json/img/yoga_eval'
# image_dir = "/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/Img2json/img/tree_new"
json_dir = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/Img2json/img/yoga_eval'


yolo_cfg = "yolo/cfg/yolov3-spp.cfg"
yolo_weight = 'models/yolo/yolov3-spp.weights'
# yolo_cfg = '../../MODELS/ceiling_detection/0825/yolov3-spp-1cls-leaky.cfg'
# yolo_weight = "../../MODELS/ceiling_detection/0825/backup40.weights"
# yolo_cfg = '../../MODELS/underwater_rgb/0902_spp/yolov3-spp-1cls-leaky.cfg'
# yolo_weight = "../../MODELS/underwater_rgb/0902_spp/best.weights"
model_path = 'models/sppe/duc_se.pth'
# model_path = '../../PoseTrainingPytorch/exp/ceiling_tmp/1/1_best.pkl'
# model_path = '../../MODELS/underwater_pose/underwater_5/underwater_5_best.pkl'


#json2h5
#0_rename_img
img_folder = "/home/hkuit155/Documents/yoga_train_new1"
dest_folder = "/home/hkuit155/Documents/yoga_train"

#1_renamejson
#ceiling pose:3
#underwater pose:2
#yoga+ai: 0
imageandjson_first_number = "1"

number_json_file = 2
json_filename = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/json2h5/json/0612_underwater_all.json'
output_jsonname = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/json2h5/json/0612_underwater_all_rename.json'

#2_json2h5_others/2_json2h5_kp2json
input_json = ["/home/hkuit155/Documents/yoga_train.json"]
output_h5name = '/home/hkuit155/Documents/yoga_train.h5'
