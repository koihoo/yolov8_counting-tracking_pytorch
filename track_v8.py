from ultralytics import YOLO

# model = YOLO("/home/koihoo/yolov8/runs/detect/train7/weights/best.pt")  # or a segmentation model .i.e yolov8n-seg.pt
model = YOLO("/home/koihoo/ultralytics/yolov8x.pt")  # or a segmentation model .i.e yolov8n-seg.pt
model.track(
    # source="/home/koihoo/yolov8/data/goose-video/TundraSwan2.mp4",
     source="/home/koihoo/ultralytics/bus.jpg",
    # stream=True,
    tracker="/home/koihoo/ultralytics/ultralytics/tracker/cfg/botsort.yaml",  # or 'bytetrack.yaml'
    save = True,
    conf = 0.6,
    hide_conf = True,
)