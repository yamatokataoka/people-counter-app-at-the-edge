"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# logging
# log.basicConfig(level=log.DEBUG)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, port=MQTT_PORT, keepalive=MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    model = args.model
    device = args.device
    cpu_extension = args.cpu_extension
    infer_network.load_model(model, device, cpu_extension)

    network_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # Get and open video capture
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    # for publishing information
    total_count = 0
    previous_count = 0
    duration = 0
    start_time = 0

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        current_count = 0

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        try:
            p_frame = cv2.resize(frame, (network_input_shape[3], network_input_shape[2]))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)
        except Exception as e:
            log.error("exception: {}".format(e))
            break

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()

            output_frame = draw_boxes(frame, result, prob_threshold, width, height)

            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            for box in result[0][0]:
                confidence = box[2]
                if confidence >= prob_threshold and box[1] == 1:
                    current_count += 1
            client.publish("person", json.dumps({"count": current_count}))
            log.info("current_count at {}: {}".format(time.time(), current_count))

            if current_count > previous_count:
                start_time = time.time()
                addition = current_count - previous_count
                total_count += addition
                log.info("total_count at {}: {}".format(time.time(),total_count))
                client.publish("person", json.dumps({"total": total_count}))
            elif current_count < previous_count:
                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))

            previous_count = current_count

        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(output_frame)
        sys.stdout.flush()

        # Break if escape key pressed
        if key_pressed == 27:
            break

        ### Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('output.jpg', frame)

        ### Close the stream and any windows at the end of the application
        cap.release()
        cv2.destroyAllWindows()

        # Disconnect from MQTT
        client.disconnect()

def draw_boxes(frame, result, prob_threshold, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]:
        confidence = box[2]
        if confidence >= prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)

            # Draw the detected bounding boxes
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
    return frame

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
