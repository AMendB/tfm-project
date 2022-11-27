#!/usr/bin/env python3

import sys
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
import torch.backends.cudnn as cudnn

sys.path.append('./yolov5')
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from threading import Lock, Thread
from time import sleep

# import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer_edited as cv_viewer

lock = Lock()
run_signal = False
exit_signal = False

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()

def img_preprocess(img, device, half, net_size):
    net_image, ratio, pad = letterbox(img[:, :, :3], net_size, auto=False)
    net_image = net_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    net_image = np.ascontiguousarray(net_image)

    img = torch.from_numpy(net_image).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, ratio, pad


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) * im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) * im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) * im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) * im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_min
    output[2][1] = y_max

    output[3][0] = x_max
    output[3][1] = y_max
    return output


def detections_to_custom_box(detections, im, im0):
    output = []
    for i, det in enumerate(detections):
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                # Creating ingestable objects for the ZED SDK
                obj = sl.CustomBoxObjectData()
                obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
                obj.label = cls
                obj.probability = conf
                obj.is_grounded = False
                output.append(obj)
    return output


def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections

    print("Intializing Network...")

    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA
    imgsz = img_size

    # Load model
    model = attempt_load(weights)  # load FP32
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    cudnn.benchmark = True

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    while not exit_signal:
        if run_signal:
            lock.acquire()
            img, ratio, pad = img_preprocess(image_net, device, half, imgsz)

            pred = model(img)[0]
            det = non_max_suppression(pred, conf_thres, iou_thres)

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, img, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)


def main():
    global image_net, exit_signal, run_signal, detections

    capture_thread = Thread(target=torch_thread,
                            kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")



    # Create a InitParameters object and set configuration parameters
    input_type = sl.InputType()
    if opt.svo is not None: #carga del svo
        input_type.set_from_svo_file(opt.svo)

    init_params = sl.InitParameters(input_t=input_type)
    # init_params.svo_real_time_mode = True
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 20
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY: recommend using the ULTRA depth mode to improve depth accuracy at long distances.
    

    # Create ZED objects
    zed = sl.Camera()
    runtime_params = sl.RuntimeParameters()
    runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD # b: Use the STANDARD mode for applications such as autonomous navigation, obstacle detection, 3D mapping, people detection and tracking.
    runtime_params.confidence_threshold = 100 # 100 por defecto
    runtime_params.texture_confidence_threshold = 70 # 100 por defecto
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()


    # Set initialization parameters
    obj_detection_params = sl.ObjectDetectionParameters()
    obj_detection_params.image_sync = True # determines if object detection runs for each frame or asynchronously in a separate thread
    obj_detection_params.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_detection_params.enable_mask_output = True # Outputs 2D masks over detected objects
    obj_detection_params.enable_tracking = False # allows objects to be tracked across frames and keep the same ID as long as possible. Positional tracking must be active in order to track objects movements independently from camera motion.
    if obj_detection_params.enable_tracking :
        # Set positional tracking parameters
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        # Enable positional tracking
        zed.enable_positional_tracking(positional_tracking_parameters)
    print("Object Detection: Loading Module...")
    err = zed.enable_object_detection(obj_detection_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error {}, exit program".format(err))
        zed.close()
        exit()


    objects = sl.Objects() # Structure containing all the detected objects
    # Configure object detection runtime parameters
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    # detection_confidence = 1
    # obj_runtime_param.detection_confidence_threshold = detection_confidence

    # Sacar left image    
    camera_info = zed.get_camera_information()
    image_left = sl.Mat()

    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
                                       min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width, display_resolution.height / camera_info.camera_resolution.height]
    
    # Utilities for tracks view
    camera_config = zed.get_camera_information().camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.camera_fps,
                                                    init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    
    # Camera pose
    cam_w_pose = sl.Pose()

    image_size = zed.get_camera_information().camera_resolution
    width = image_size.width
    height = image_size.height

    # Prepare side by side image container equivalent to CV_8UC4
    width_sbs = width*2+tracks_resolution.width
    svo_image_sbs_rgb = np.zeros((height, width_sbs, 3), dtype=np.uint8)

    # Prepare single image containers
    image_left_tmp = sl.Mat()
    image_depth = sl.Mat()

    # Create video writer with MPEG-4 part 2 codec
    video_writer = cv2.VideoWriter(opt.output_path,
                                    cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                    max(zed.get_camera_information().camera_fps, 25),
                                    (width_sbs, height))

    if not video_writer.isOpened():
        sys.stdout.write("OpenCV video writer cannot be opened. Please check the .avi file path and write "
                            "permissions.\n")
        zed.close()
        exit()

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")
    nb_frames = zed.get_svo_number_of_frames()

    while not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            zed.retrieve_image(image_depth, sl.VIEW.DEPTH) # saco imagen profundidad del svo
            image_net = image_left_tmp.get_data() # obtengo los datos de la imagen izq para un formato válido para la net de detección
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param) # obtiene vectores con toda la información de los objetos detectados: distancia, posición de los boxes...

            # -- Display
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution) 
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 2D rendering
            image_left_ocv = image_left.get_data() # copia de la imagen en formato numpy
            image_depth_ocv = image_depth.get_data() # copia de la imagen en formato numpy

            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_detection_params.enable_tracking) # pinta las bounding boxes en la imagen izq
            cv_viewer.render_2D(image_depth_ocv, image_scale, objects, obj_detection_params.enable_tracking) # pinta las bounding boxes en la imagen de profundidad
            global_image = cv2.hconcat([image_left_ocv, image_depth_ocv,image_track_ocv])

            # Tracking view
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)
            
            if opt.dontshow:
                pass
            else:
                cv2.imshow("ZED | 2D View and Birds View", global_image)
                key = cv2.waitKey(1)
                if key == 27:
                    exit_signal = True

            # Convert SVO image from RGBA to RGB
            global_image = cv2.cvtColor(global_image, cv2.COLOR_RGBA2RGB)
            
            # Write the RGB image in the video
            video_writer.write(global_image)

            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
        else:
            exit_signal = True
    exit_signal = True
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--output_path', type=str, default='output_detection.avi', help='video name')
    parser.add_argument('--dontshow', action='store_true', default=False, help='flag to not show frames while running')
    opt = parser.parse_args()

    with torch.no_grad():
        main()