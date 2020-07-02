from Img2json.estimator.pose_estimator import PoseEstimator
from Img2json.detector.yolo_detect import ObjectDetectionYolo
from Img2json.config import config
import torch
import cv2
import copy
import os


class ImageFolderDetection:
    def __init__(self, src_folder, dest_folder):
        self.src_img_ls = [os.path.join(src_folder, img_name) for img_name in os.listdir(src_folder)]
        self.dest_img_ls = [os.path.join(dest_folder, img_name) for img_name in os.listdir(src_folder)]
        self.object_detector = ObjectDetectionYolo()
        self.pose_estimator = PoseEstimator()
        self.json_file = open("img/tree.txt", "w")
        self.idx = 0

    def __process_img(self, img_path, dest_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, config.frame_size)    # Delete if not required to resize
        with torch.no_grad():
            inps, orig_img, boxes, scores, pt1, pt2 = self.object_detector.process(img)
            if boxes is not None:
                key_points, self.img, self.img_black = self.pose_estimator.process_img(inps, orig_img, boxes, scores,
                                                                                       pt1, pt2)
                if key_points:
                    cv2.imwrite(dest_path, self.img)
                    cv2.imshow("res", self.img)
                    cv2.waitKey(2)

                    self.__write_json()

    def __write_json(self):
        self.json_file.write(str(self.idx)+'\t')

    def process(self):
        for self.idx in range(len(self.src_img_ls)):
            print("Processing image {}".format(self.idx))
            self.__process_img(self.src_img_ls[self.idx], self.dest_img_ls[self.idx])


if __name__ == '__main__':
    src = "img/tree"
    dest = "img/tree_kps"
    os.makedirs(dest, exist_ok=True)
    IFD = ImageFolderDetection(src, dest)
    IFD.process()
