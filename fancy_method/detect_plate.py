# basic method
import torch
import os
import shutil
import random
import time
import cv2
from pathlib import Path
import sys
# utils method
import utils.torch_utils as torch_utils
import utils.google_utils as google_utils
import utils.common_utils as common_utils

from utils.datasets import LoadStreams, LoadImages


def detect(save_img=False, **kwargs):
    """

    :param save_img:
    :param kwargs: 参数包，详见下方解释
    :return:
    """
    # 参数捕获
    print(f'-- kwargs -- {kwargs}')
    out = kwargs.get('output', 'inference/output')  # Str: 输出路径
    source = kwargs.get('source', 'data/images/test_BW')  # Str: 检测路径 文件夹
    weights = kwargs.get('weights', 'weights/last.pt')  # Str: 最终训练模型系数
    save_txt = kwargs.get('save_txt', False)  # Bool: 是否保存文档
    imgsz = kwargs.get('imgsz', 640)  # Int: 图像大小
    device = kwargs.get('device', '')  # Str: 使用cpu/gpu 详见select_device函数
    iou_thres = kwargs.get('iou_thres', 0.5)  # Float: 交叉占比限制，超过限制的交叉方框删除
    conf_thres = kwargs.get('conf_thres', 0.4)  # Float: 筛选方框置信度
    classes = kwargs.get('classes', list())  # List: 仅预测特定类，传list
    augment = kwargs.get('augment', False)  # Bool
    agnostic_nms = kwargs.get('agnostic_nms', False)  # Bool: 详见详见non_max_suppression
    fourcc = kwargs.get('fourcc', 'mp4v')
    view_img = kwargs.get('view_img', False)  # 是否查看图片

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    # Initialize pytorch use yaml file
    device = torch_utils.select_device(device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    #######
    ckpt = torch.load(weights, map_location=device)
    model = ckpt['model'].float()
    #######
    # model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    res = list()
    for path, img, im0s, vid_cap in dataset:
        tmp = {'file_path': path, 'detect_box': list()}
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = common_utils.non_max_suppression(pred, conf_thres, iou_thres,
                                                fast=True, classes=classes, agnostic=agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = common_utils.apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = common_utils.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (common_utils.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        common_utils.plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    tmp['detect_box'].append(xywh)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imencode('.jpg', im0)[1].tofile(save_path)
                    # cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
        res.append(tmp)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if sys.platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    print('Done. (%.3fs)' % (time.time() - t0))
    return res


def detect_main(*args, **kwargs):
    # 检查、调整图像大小
    pass
    # 运行
    with torch.no_grad():
        res = detect(*args, **kwargs)
    return res


if __name__ == '__main__':
    detect_main()
