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
        self.src_img_ls = [os.path.join(src_folder, img_name) for img_name in os.listdir(src_folder)]
        self.dest_img_ls = [os.path.join(dest_folder, img_name) for img_name in os.listdir(src_folder)]
        self.idx = 0
        self.keypoints_json = []
        self.bbox = []
        self.result = {}
        self.result_all = {}
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
        self.boxes = [[]]
        self.key_points = [[]]
        self.kps_score = [[]]

    def clear(self):
        self.boxes = [[]]
        self.key_points = [[]]
        self.kps_score = [[]]

    def __process_img(self, img_path, dest_path):
        self.img = cv2.imread(img_path)
        # if not img:
        #     return
        # try:
        with torch.no_grad():
            try:
                inps, orig_img, boxes, scores, pt1, pt2 = object_detector.process(self.img)
                if boxes is not None:
                    self.boxes = torch.Tensor([[item[0],item[1],item[2]-item[0],item[3]-item[1]] for item in boxes.tolist()])
                    self.key_points, self.img, self.img_black, self.kps_score = pose_estimator.process_img(inps, orig_img, boxes, scores,
                                                                                           pt1, pt2)
            except ValueError:
                print(img_path)
                pass
                # if key_points:
            cv2.imwrite(dest_path, self.img)
            self.writeJson(self.img, self.key_points, self.boxes)
            self.img = cv2.resize(self.img, (720, 540))
            cv2.imshow("res", self.img)
            cv2.waitKey(2)

    # def writeImageJson(self, image):


    def writeJson(self, image, kps, bbox):
        file_name = str(self.src_img_ls[self.idx]).split("/")[-1]
        id = self.id_cnt
        width = image.shape[1]
        height = image.shape[0]
        url = "http://localhost:8007/" + file_name
        # # annotation data
        image_id = self.id_cnt
        some_id = image_id

        self.images.append({"id": id,
            "file_name": file_name,
                            "width": width,
                            "height": height,
                            "url": url})
        # except:
        #     pass

    def writeJson(self, image, kps, bbox):
        file_name = str(self.src_img_ls[self.idx]).split("/")[-1]
        id = self.id_cnt
        width = image.shape[1]
        height = image.shape[0]
        url = "http://localhost:8007/" + file_name
        # annotation data
        image_id = self.id_cnt
        some_id = image_id

        self.images.append({"id": id,
                            "file_name": file_name,
                            "width": width,
                            "height": height,
                            "url": url,
                            "image_id": file_name,
                            "width": width,
                            "height": height,
                            "url": url})
        # # print("id: ", self.id_cnt, ", length: ", len(kps[0]))

        if len(bbox) > 0 and len(kps) > 0:
            for i in range(len(kps[0])):
                for j in range(len(kps[0][0])):
                    self.keypoints_json.append(kps[0][i][j].item())
                if self.kps_score[0][i] > 0.3:
                    self.keypoints_json.append(2)
                else:
                    self.keypoints_json.append(0)
            for j in range(4):
                self.bbox.append(bbox[0][j].item())

            self.annotations.append({"image_name": file_name,
                                     "category_id": 0,
                                     "bbox": self.bbox,
                                     "keypoints": self.keypoints_json,
                                     "id": some_id})

        self.id_cnt += 1
        self.keypoints_json = []
        self.bbox = []

    def process(self):
        for self.idx in range(len(self.src_img_ls)):
            print("Processing image {}".format(self.idx))

            # print("Processing image {}".format(self.src_img_ls[self.idx]))
            try:
                self.__process_img(self.src_img_ls[self.idx], self.dest_img_ls[self.idx])
            except ValueError:
                print(self.src_img_ls[self.idx][-16:])

        self.result_all["images"] = self.images
        self.result_all["annotations"] = self.annotations
        self.json.write(json.dumps(self.result_all))


if __name__ == '__main__':
    src_folder = config.image_dir
    dest_folder = src_folder + "_kps"
    os.makedirs(dest_folder, exist_ok=True)
    ImageDetection(src_folder, dest_folder).process()

