import numpy as np

"""produce anchors"""


class Anchor_ms(object):
    """
    stable version for anchor generator
    """

    def __init__(self, feature_w, feature_h):
        self.w = feature_w
        self.h = feature_h
        self.base = 64  # base size for anchor box
        self.stride = 15  # center point shift stride
        self.scale = [1 / 3, 1 / 2, 1, 2, 3]  # aspect ratio
        self.anchors = self.gen_anchors()  # xywh
        self.eps = 0.01

    def gen_single_anchor(self):
        scale = np.array(self.scale, dtype=np.float32)
        s = self.base * self.base
        w, h = np.sqrt(s / scale), np.sqrt(s * scale)
        c_x, c_y = (self.stride - 1) // 2, (self.stride - 1) // 2
        anchor = np.vstack([c_x * np.ones_like(scale, dtype=np.float32), c_y * np.ones_like(scale, dtype=np.float32), w,
                            h]).transpose()
        anchor = self.center_to_corner(anchor)
        return anchor

    def gen_anchors(self):
        anchor = self.gen_single_anchor()
        k = anchor.shape[0]
        delta_x, delta_y = [x * self.stride for x in range(self.w)], [y * self.stride for y in range(self.h)]
        shift_x, shift_y = np.meshgrid(delta_x, delta_y)
        shifts = np.vstack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()]).transpose()
        a = shifts.shape[0]
        anchors = (anchor.reshape((1, k, 4)) + shifts.reshape((a, 1, 4))).reshape((a * k, 4))  # corner forma
        anchors = self.corner_to_center(anchors)
        return anchors

    # float,中点变四角
    def center_to_corner(self, box):
        box = box.copy()
        box_ = np.zeros_like(box, dtype=np.float32)
        box_[:, 0] = box[:, 0] - (box[:, 2] - 1) / 2
        box_[:, 1] = box[:, 1] - (box[:, 3] - 1) / 2
        box_[:, 2] = box[:, 0] + (box[:, 2] - 1) / 2
        box_[:, 3] = box[:, 1] + (box[:, 3] - 1) / 2
        box_ = box_.astype(np.float32)
        return box_

    # float，四角变中点，x,y,w,h
    def corner_to_center(self, box):
        box = box.copy()
        box_ = np.zeros_like(box, dtype=np.float32)
        box_[:, 0] = box[:, 0] + (box[:, 2] - box[:, 0]) / 2
        box_[:, 1] = box[:, 1] + (box[:, 3] - box[:, 1]) / 2
        box_[:, 2] = (box[:, 2] - box[:, 0])
        box_[:, 3] = (box[:, 3] - box[:, 1])
        box_ = box_.astype(np.float32)
        return box_

    def pos_neg_anchor(self, gt, pos_num=16, neg_num=48, threshold_pos=0.5, threshold_neg=0.1):
        gt = gt.copy()
        gt_corner = self.center_to_corner(np.array(gt, dtype=np.float32).reshape(1, 4))
        an_corner = self.center_to_corner(np.array(self.anchors, dtype=np.float32))
        iou_value = self.iou(an_corner, gt_corner).reshape(-1)  # (1445)
        max_iou = max(iou_value)
        pos, neg = np.zeros_like(iou_value, dtype=np.int32), np.zeros_like(iou_value, dtype=np.int32)

        # pos
        pos_cand = np.argsort(iou_value)[::-1][:30]
        pos_index = np.random.choice(pos_cand, pos_num, replace=False)
        if max_iou > threshold_pos:
            pos[pos_index] = 1

        # neg
        neg_cand = np.where(iou_value < threshold_neg)[0]
        neg_ind = np.random.choice(neg_cand, neg_num, replace=False)
        neg[neg_ind] = 1

        return pos, neg

    # float
    def diff_anchor_gt(self, gt):
        eps = self.eps
        anchors, gt = self.anchors.copy(), gt.copy()
        diff = np.zeros_like(anchors, dtype=np.float32)
        diff[:, 0] = (gt[0] - anchors[:, 0]) / (anchors[:, 2] + eps)
        diff[:, 1] = (gt[1] - anchors[:, 1]) / (anchors[:, 3] + eps)
        diff[:, 2] = np.log((gt[2] + eps) / (anchors[:, 2] + eps))
        diff[:, 3] = np.log((gt[3] + eps) / (anchors[:, 3] + eps))
        return diff
    def iou(self, box1, box2):
        box1, box2 = box1.copy(), box2.copy()
        N = box1.shape[0]
        K = box2.shape[0]
        box1 = np.array(box1.reshape((N, 1, 4))) + np.zeros((1, K, 4))  # box1=[N,K,4]
        box2 = np.array(box2.reshape((1, K, 4))) + np.zeros((N, 1, 4))  # box1=[N,K,4]
        x_max = np.max(np.stack((box1[:, :, 0], box2[:, :, 0]), axis=-1), axis=2)
        x_min = np.min(np.stack((box1[:, :, 2], box2[:, :, 2]), axis=-1), axis=2)
        y_max = np.max(np.stack((box1[:, :, 1], box2[:, :, 1]), axis=-1), axis=2)
        y_min = np.min(np.stack((box1[:, :, 3], box2[:, :, 3]), axis=-1), axis=2)
        tb = x_min - x_max
        lr = y_min - y_max
        tb[np.where(tb < 0)] = 0
        lr[np.where(lr < 0)] = 0
        over_square = tb * lr
        all_square = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1]) + (
                box2[:, :, 2] - box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1]) - over_square
        return over_square / all_square