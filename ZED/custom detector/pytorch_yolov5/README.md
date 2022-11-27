# Object Detection using ZED SDK and YOLOv5

This sample shows how to detect custom objects using the official Pytorch implementation of YOLOv5 from a ZED camera and ingest them into the ZED SDK to extract 3D informations and tracking for each objects.

## Getting started

 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Clone Yolov5 into the current folder

```sh
git clone https://github.com/ultralytics/yolov5
# Install the dependencies if needed
cd yolov5
pip install -r requirements.txt
```

- Download the model file 

## Run the program

```
python detector.py --svo path/to/file.svo --weights best.pt --conf_thres 0.6 --output_path path/to/output_file.avi --dontshow
```

### Features

 - Bounding boxes around detected objects are drawn, with class and depth measure
 - Objects classes and confidences can be changed

