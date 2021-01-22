from Img2json.estimator.pose_estimator import PoseEstimator
from Img2json.detector.yolo_detect_img import ObjectDetectionYolo
import torch
import cv2
import os
import json
import config

object_detector = ObjectDetectionYolo()
pose_estimator = PoseEstimator()



class ImageDetection:
    def __init__(self, src_folder, dest_folder):
        self.src_img_ls = [os.path.join(src_folder, img_name).replace("\\", "/") for img_name in os.listdir(src_folder)]
        self.dest_img_ls = [os.path.join(dest_folder, img_name) for img_name in os.listdir(src_folder)]
        self.idx = 0
        self.keypoints_json = []
        self.bbox = []
        self.result = {}
        self.result_all = {}
        self.person_id = 0
        self.result_all['images'] = []
        self.json = open(src_folder+".json", "w")

        self.result = {}
        self.keypoints = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"
        ]
        self.keypoints_style = [
            "#FF0000",
            "#FF7000",
            "#FFFF00",
            "#99FF00",
            "#00FF00",
            "#00FF99",
            "#00FFFF",
            "#0099FF",
            "#0000FF",
            "#9900FF",
            "#FF00FF",
            "#FF0099",
            "#FFAAAA",
            "#FFCCAA",
            "#FFFFAA",
            "#AAFFAA",
            "#AAFFFF"
        ]
        self.categories = [{
            "id": "0",
            "name": "human",
            "supercategory": "human",
            "keypoints": self.keypoints,
            "keypoints_style": self.keypoints_style
        }]
        self.licenses = []
        self.images = []
        self.annotations = []
        self.id_cnt = 0
        self.result_all["categories"] = self.categories
        self.result_all["licenses"] = self.licenses

    def clear(self):
        pass

    def __process_img(self, img_path, dest_path):
        self.clear()

        frame = cv2.imread(img_path)
        self.writeImageJson(frame)
        with torch.no_grad():
            inps, orig_img, boxes, scores, pt1, pt2 = object_detector.process(frame)
            if boxes is not None:
                key_points, self.img, self.img_black, score = pose_estimator.process_img(inps, orig_img, boxes, scores,
                                                                                         pt1, pt2)
                cv2.imwrite(dest_path, frame)
                self.writeJson(key_points, boxes, score)
                img = cv2.resize(frame, (720, 540))
                cv2.imshow("res", img)
                cv2.waitKey(2)
                self.id_cnt += 1

    def writeImageJson(self, image):
        file_name = str(self.src_img_ls[self.idx]).split("/")[-1]
        width, height = image.shape[1], image.shape[0]
        url = "http://localhost:8007/" + file_name
        # # annotation data

        self.images.append({"id": self.id_cnt,
                            "file_name": file_name,
                            "width": width,
                            "height": height,
                            "url": url})

    def writeJson(self, kps_dict, bbox_dict, kpScore_dict):
        keypoints_json = []
        box, kps, kpScore = bbox_dict, kps_dict, kpScore_dict
        if len(box) > 0 and len(kps) > 0:
            for i in range(len(kps[0])):
                for j in range(len(kps[0][0])):
                    keypoints_json.append(kps[0][i][j].item())
                if kpScore[0][i] > 0.3:
                    keypoints_json.append(2)
                else:
                    keypoints_json.append(0)
            # for j in range(4):
            #     bbox.append(box[j].item())
            w, h = box[0][2] - box[0][0], box[0][3] - box[0][1]
            box_tmp = [box[0][0].tolist(), box[0][1].tolist(), w.tolist(), h.tolist()]
            self.annotations.append({"image_id": str(self.id_cnt),
                                     "category_id": "0",
                                     "bbox": box_tmp,
                                     "keypoints": keypoints_json,
                                     "id": str(self.person_id)})
        self.person_id += 1

    def process(self):
        for self.idx, (src, dest) in enumerate(zip(self.src_img_ls, self.dest_img_ls)):
            self.__process_img(src, dest)

        self.result_all["images"] = self.images
        self.result_all["annotations"] = self.annotations
        self.json.write(json.dumps(self.result_all))


if __name__ == '__main__':
    src_folder = 'img/tree'
    dest_folder = src_folder + "_kps"
    os.makedirs(dest_folder, exist_ok=True)
    ImageDetection(src_folder, dest_folder).process()