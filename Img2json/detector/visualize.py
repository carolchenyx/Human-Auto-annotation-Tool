import cv2


class BBoxVisualizer(object):
    def __init__(self):
        self.color = (0, 0, 255)

    def visualize(self, bboxes, img):
        for bbox in bboxes:
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.color, 2)
        return img