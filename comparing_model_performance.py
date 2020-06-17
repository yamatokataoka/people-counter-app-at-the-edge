import os
import six.moves.urllib as urllib
import sys
import tarfile
import time
import tensorflow as tf
import cv2

import logging as log
from inference import Network

log.basicConfig(level=log.DEBUG)

VIDEO_PATH = "./resources/Pedestrian_Detect_2_1_1.mp4"
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# load tensorflow model
def load_model():
    # Download Model
    if not os.path.exists(os.path.join(os.getcwd(), MODEL_FILE)):
        log.info("Downloading model")
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph

def infer_on_stream():
    # Initialise the class
    infer_network = Network()

    ### Load the model using openvino ###
    # model = "./intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml"
    model = "./frozen_inference_graph.xml"
    device = "CPU"
    cpu_extension = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    infer_network.load_model(model, device, cpu_extension)
    
    openvino_input_shape = infer_network.get_input_shape()

    ### Load the model using tensorflow ###
    detection_graph = load_model()

    # Get and open video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.open(VIDEO_PATH)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # time performance
    start_time = time.time()

    ### Tensorflow: Loop until stream is over ###
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            while cap.isOpened():
                
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
                if current_frame % 50 == 0:
                    log.info("current frame: {}/{}".format(current_frame, total_frames))
                
                ### Read from the video capture ###
                flag, frame = cap.read()
                if not flag:
                    break

                # Extract tensors
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                ### Pre-process the image as needed ###
                try:
                    p_frame = cv2.resize(frame, (openvino_input_shape[3], openvino_input_shape[2]))
                    p_frame = p_frame.reshape(1, *p_frame.shape)
                except Exception as e:
                    log.error("exception: {}".format(e))
                    break

                ### Run inference ###
                (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: p_frame})

    end_time = time.time()

    log.info("Tensorflow: {}".format(end_time - start_time))

    # reset frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    log.info("current frame after restting: {}/{}".format(cap.get(cv2.CAP_PROP_POS_FRAMES), total_frames))

    # time performance
    start_time = time.time()

    ### OpenVINO: Loop until stream is over ###
    while cap.isOpened():

        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        if current_frame % 50 == 0:
            log.info("current frame: {}/{}".format(current_frame, total_frames))
        
        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### Pre-process the image as needed ###
        try:
            p_frame = cv2.resize(frame, (openvino_input_shape[3], openvino_input_shape[2]))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)
        except Exception as e:
            log.error("exception: {}".format(e))
            break

        ### Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)
        infer_network.wait()

    end_time = time.time()
    
    log.info("OpenVINO: {}".format(end_time - start_time))


def main():
    """
    Compare model performance

    :return: None
    """
    # Perform inference on the input stream
    infer_on_stream()


if __name__ == '__main__':
    main()