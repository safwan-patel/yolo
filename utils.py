def bbox_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersect_w = abs(max(x1, x2) - min(x1 + w1, x2 + w2))
    intersect_h = abs(max(y1, y2) - min(y1 + h1, y2 + h2))
    
    intersect = intersect_w * intersect_h
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union