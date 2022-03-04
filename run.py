# 创建应用实例
import sys

from wxcloudrun import app

from flask import request
import json
from cv2 import cv2
import torch


# 只接受POST方法访问
@app.route("/", methods=["POST", "GET"])
def check():
    # 默认返回内容
    model = torch.hub.load('./YOLOV5-5.0/yolov5-5.0/yolov5-5.0', 'custom',
                           path_or_model='./YOLOV5-5.0/yolov5-5.0/yolov5-5.0/yolov5s.pt',
                           source='local')
    return_dict = {"result": ""}
    name_list = []
    confidence_list = []
    # 判断入参是否为空
    if request.get_data() is None:
        return json.dumps(return_dict, ensure_ascii=False)
    # print(request.files)
    # print(request.values)
    # print(request.data)
    img_data = request.files['file']
    # print(img_data)
    file_path = "./ok.jpg"
    img_data.save(file_path)
    imgs = cv2.imread('./ok.jpg')
    # Inference
    results = model(imgs, size=640)  # includes NMS
    res_data = results.pandas().xyxy[0]

    name = res_data['name']
    for name_value in name:
        name_list.append(name_value)
    confidence = res_data['confidence']
    for confidence_value in confidence:
        confidence_list.append(confidence_value)

    idx = confidence_list.index(max(confidence_list))
    return_name = name_list[idx]
    return_confi = confidence_list[idx]
    return_dict["result"] = "%s:%s" % (return_name, return_confi)
    return json.dumps(return_dict, ensure_ascii=False)


# 启动Flask Web服务
if __name__ == '__main__':
    app.run(host=sys.argv[1], port=sys.argv[2])
