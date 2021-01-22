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


    def __process_img(self, img_path, dest_path):
        img = cv2.imread(img_path)
        with torch.no_grad():
            inps, orig_img, boxes, scores, pt1, pt2 = object_detector.process(img)
            if boxes is not None:
                boxes = torch.Tensor([[item[0],item[1],item[2]-item[0],item[3]-item[1]] for item in boxes.tolist()])

                key_points, self.img, self.img_black, _ = pose_estimator.process_img(inps, orig_img, boxes, scores,

                                                                                       pt1, pt2)
                if key_points:
                    cv2.imwrite(dest_path, self.img)
                    cv2.imshow("res", self.img)
                    cv2.waitKey(2)
                    self.__write_json(img, key_points,boxes)

    def __write_json(self, image, keypoints,boxes):
        for i in range(len(keypoints[0])):
            for j in range(len(keypoints[0][0])):
                self.keypoints_json.append(keypoints[0][i][j].item())
        self.result['image_name'] = str(self.src_img_ls[self.idx])[-16:]
        for j in range(4):
            self.bbox.append(boxes[0][j].item())
        self.result['bbox'] = self.bbox
        self.result['id'] = self.idx
        self.result['height'] = image.shape[0]
        self.result['width'] = image.shape[1]
        self.result['keypoints'] = self.keypoints_json
        self.result_all['images'].append( self.result)
        self.result = {}
        self.keypoints_json = []
        self.bbox = []

    def process(self):
        for self.idx in range(len(self.src_img_ls)):
            print("Processing image {}".format(self.idx))
            try:
                self.__process_img(self.src_img_ls[self.idx], self.dest_img_ls[self.idx])
            except ValueError:
                print(self.src_img_ls[self.idx][-16:])

        self.json.write(json.dumps(self.result_all))




if __name__ == '__main__':

    src_folder = config.image_dir
    dest_folder = src_folder + "_kps"
    os.makedirs(dest_folder, exist_ok=True)
    ImageDetection(src_folder, dest_folder).process()
