import torch
from Img2json.SPPE.src.utils.img import cropBox, im_to_torch
from Img2json.config import config
from Img2json.yolo.preprocess import prep_frame
from Img2json.yolo.util import dynamic_write_results
from Img2json.yolo.darknet import Darknet
import cv2
from Img2json.config import config
import config as config_de



yolo_cfg = config_de.yolo_cfg
yolo_weight = config_de.yolo_weight



class ObjectDetectionYolo(object):
    def __init__(self, batchSize=1):
        self.det_model = Darknet(yolo_cfg)
        self.det_model.load_weights(yolo_weight)
        self.det_model.net_info['height'] = config.input_size
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()

        self.stopped = False
        self.batchSize = batchSize

    def __video_process(self, frame):
        img = []
        orig_img = []
        im_name = []
        im_dim_list = []
        img_k, orig_img_k, im_dim_list_k = prep_frame(frame, int(config.input_size))

        img.append(img_k)
        orig_img.append(orig_img_k)
        im_name.append('0.jpg')
        im_dim_list.append(im_dim_list_k)

        with torch.no_grad():
            # Human Detection
            img = torch.cat(img)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
        return img, orig_img, im_name, im_dim_list

    def __get_bbox(self, img, orig_img, im_name, im_dim_list):
        with torch.no_grad():
            # Human Detection
            img = img.cuda()
            prediction = self.det_model(img, CUDA=True)
            # NMS process
            dets = dynamic_write_results(prediction, config.confidence,  config.num_classes, nms=True, nms_conf=config.nms_thresh)

            if isinstance(dets, int) or dets.shape[0] == 0:
                return orig_img[0], im_name[0], None, None, None, None, None

            dets = dets.cpu()
            im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

            # coordinate transfer
            dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

            dets[:, 1:5] /= scaling_factor
            for j in range(dets.shape[0]):
                dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]

        boxes_k = boxes[dets[:, 0] == 0]
        if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
            return orig_img[0], im_name[0], None, None, None, None, None
        inps = torch.zeros(boxes_k.size(0), 3, config.input_height, config.input_width)
        pt1 = torch.zeros(boxes_k.size(0), 2)
        pt2 = torch.zeros(boxes_k.size(0), 2)
        return orig_img[0], im_name[0], boxes_k, scores[dets[:, 0] == 0], inps, pt1, pt2

    def __crop_bbox(self, orig_img, im_name, boxes, scores, inps, pt1, pt2):
        with torch.no_grad():
            if orig_img is None:
                return None, None, None, None, None, None, None

            if boxes is None or boxes.nelement() == 0:
                return None, orig_img, im_name, boxes, scores, None, None

            inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = self.__crop_from_dets(inp, boxes, inps, pt1, pt2)
            return inps, orig_img, im_name, boxes, scores, pt1, pt2

    @staticmethod
    def __crop_from_dets(img, boxes, inps, pt1, pt2):
        '''
        Crop human from origin image according to Dectecion Results
        '''

        imght = img.size(1)
        imgwidth = img.size(2)
        tmp_img = img
        tmp_img[0].add_(-0.406)
        tmp_img[1].add_(-0.457)
        tmp_img[2].add_(-0.480)
        for i, box in enumerate(boxes):
            upLeft = torch.Tensor((float(box[0]), float(box[1])))
            bottomRight = torch.Tensor((float(box[2]), float(box[3])))

            ht = bottomRight[1] - upLeft[1]
            width = bottomRight[0] - upLeft[0]

            scaleRate = 0.3

            upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
            upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
            bottomRight[0] = max(
                min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
            bottomRight[1] = max(
                min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

            try:
                inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, config.input_height, config.input_width)
            except IndexError:
                print(tmp_img.shape)
                print(upLeft)
                print(bottomRight)
                print('===')
            pt1[i] = upLeft
            pt2[i] = bottomRight
        return inps, pt1, pt2

    def process(self, frame):
        img, orig_img, im_name, im_dim_list = self.__video_process(frame)
        inps, orig_img, im_name, boxes, scores, pt1, pt2 = self.__get_bbox(img, orig_img, im_name, im_dim_list)
        inps, orig_img, im_name, boxes, scores, pt1, pt2 = self.__crop_bbox(inps, orig_img, im_name,
                                                                                            boxes, scores, pt1, pt2)
        return inps, orig_img, boxes, scores, pt1, pt2

