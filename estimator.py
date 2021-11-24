import collections
import time

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from openvino import inference_engine as ie

from decoder import OpenPoseDecoder
from player import VideoPlayer

import imutils

colors = (
    (255, 0, 0),
    (255, 0, 255),
    (170, 0, 255),
    (255, 0, 85),
    (255, 0, 170),
    (85, 255, 0),
    (255, 170, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 255, 85),
    (170, 255, 0),
    (0, 85, 255),
    (0, 255, 170),
    (0, 0, 255),
    (0, 255, 255),
    (85, 0, 255),
    (0, 170, 255),
)

default_skeleton = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


class PoseEstimator(object):
    def __init__(self, device_name="AUTO", precision="FP16-INT8", video_url=""):
        hp_model_path = f"models/hp/{precision}/human-pose-estimation-0001.xml"
        hp_model_weights_path = f"models/hp/{precision}/human-pose-estimation-0001.bin"
        pc_model_path = f"models/pd/{precision}/person-detection-retail-0013.xml"
        pc_model_weights_path = f"models/pd/{precision}/person-detection-retail-0013.bin"

        # initialize inference engine
        ie_core = ie.IECore()
        # read the network and corresponding weights from file for pose estimation
        hp_net = ie_core.read_network(
            model=hp_model_path, weights=hp_model_weights_path
        )
        # load the model on the CPU (you can use GPU or MYRIAD as well)
        self.hp_exec_net = ie_core.load_network(hp_net, device_name)

        # get input and output names of nodes
        self.hp_input_key = list(self.hp_exec_net.input_info)[0]
        self.hp_output_keys = list(self.hp_exec_net.outputs.keys())

        # get input size
        self.height, self.width = self.hp_exec_net.input_info[
            self.hp_input_key
        ].tensor_desc.dims[2:]

        # read the network and corresponding weights from file for people counter
        pc_net = ie_core.read_network(
            model=pc_model_path, weights=pc_model_weights_path
        )
        # load the model on the CPU (you can use GPU or MYRIAD as well)
        self.pc_exec_net = ie_core.load_network(pc_net, device_name)

        # get input and output names of nodes
        self.pc_input_key = list(self.pc_exec_net.input_info)[0]
        self.pc_output_keys = list(self.pc_exec_net.outputs.keys())

        # get input size
        self.pc_height, self.pc_width = self.pc_exec_net.input_info[
            self.pc_input_key
        ].tensor_desc.dims[2:]

        if video_url == "":
            source = 0
        else:
            source = video_url

        self.player = VideoPlayer(
            source=source, flip=False, fps=30, skip_first_frames=0
        )
        self.decoder = OpenPoseDecoder()

        # start capturing
        self.player.start()

    def __del__(self):
        self.player.stop()

    def get_frame(self, jpeg_encoding=False):
        frame, scores = self._run_estimation()
        # _, counter = self.ssd_out()
        if jpeg_encoding:
            _, encoded_img = cv2.imencode(
                ".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            return encoded_img.tobytes(), scores
        else:
            return frame, scores

    def _pool2d(self, A, kernel_size, stride, padding, pool_mode="max"):
        # Padding
        A = np.pad(A, padding, mode="constant")

        # Window view of A
        output_shape = (
            (A.shape[0] - kernel_size) // stride + 1,
            (A.shape[1] - kernel_size) // stride + 1,
        )
        kernel_size = (kernel_size, kernel_size)
        A_w = as_strided(
            A,
            shape=output_shape + kernel_size,
            strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides,
        )
        A_w = A_w.reshape(-1, *kernel_size)

        # Return the result of pooling
        if pool_mode == "max":
            return A_w.max(axis=(1, 2)).reshape(output_shape)
        elif pool_mode == "avg":
            return A_w.mean(axis=(1, 2)).reshape(output_shape)

    # non maximum suppression
    def _heatmap_nms(self, heatmaps, pooled_heatmaps):
        return heatmaps * (heatmaps == pooled_heatmaps)

    # get poses from results
    def _process_human_pose_results(self, img, results):
        pafs = results[self.hp_output_keys[0]]
        heatmaps = results[self.hp_output_keys[1]]

        pooled_heatmaps = np.array(
            [
                [
                    self._pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max")
                    for h in heatmaps[0]
                ]
            ]
        )
        nms_heatmaps = self._heatmap_nms(heatmaps, pooled_heatmaps)

        # decode poses
        poses, scores = self.decoder(heatmaps, nms_heatmaps, pafs)
        output_shape = self.hp_exec_net.outputs[self.hp_output_keys[0]].shape
        output_scale = img.shape[1] / output_shape[3], img.shape[0] / output_shape[2]
        # multiply coordinates by scaling factor
        poses[:, :, :2] *= output_scale

        return poses, scores

    def _process_people_counter_results(self, result):
        current_count = 0
        for obj in result["detection_out"][0][0]:
            if obj[2] > 0.4:
                current_count = current_count + 1
        return current_count

    def _draw_poses(self, img, poses, point_score_threshold, skeleton=default_skeleton):
        if poses.size == 0:
            return img

        img_limbs = np.copy(img)
        pose_counter = 1
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            # Draw joints.
            joint_counter = 0
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > point_score_threshold:
                    cv2.circle(img, tuple(p), 1, colors[i], 2)
                    if joint_counter == 0:
                        cv2.putText(
                            img_limbs,
                            f"{pose_counter}",
                            tuple(p),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 255, 0),
                            3,
                            cv2.LINE_AA,
                        )
                        joint_counter += 1
            # Draw limbs.
            for i, j in skeleton:
                if (
                    points_scores[i] > point_score_threshold
                    and points_scores[j] > point_score_threshold
                ):
                    cv2.line(
                        img_limbs,
                        tuple(points[i]),
                        tuple(points[j]),
                        color=colors[j],
                        thickness=4,
                    )
            pose_counter += 1
        cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
        return img

    def _run_estimation(self):
        processing_times = collections.deque()
        # grab the frame
        frame = self.player.next()
        # if frame larger than full HD, reduce size to improve the performance
        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        # resize image and change dims to fit neural network input
        input_img = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        # create batch of images (size = 1)
        input_img = input_img.transpose(2, 0, 1)[np.newaxis, ...]

        # measure processing time
        start_time = time.time()
        # get results
        results = self.hp_exec_net.infer(inputs={self.hp_input_key: input_img})

        pc_input_img = cv2.resize(
            frame, (self.pc_width, self.pc_height), interpolation=cv2.INTER_AREA
        )
        # create batch of images (size = 1)
        pc_input_img = pc_input_img.transpose(2, 0, 1)[np.newaxis, ...]

        pc_results = self.pc_exec_net.infer(inputs={self.pc_input_key: pc_input_img})

        stop_time = time.time()
        # get poses from network results
        people_counter = self._process_people_counter_results(pc_results)
        poses, scores = self._process_human_pose_results(frame, results)

        # draw poses on a frame
        frame = self._draw_poses(frame, poses, 0.1)

        processing_times.append(stop_time - start_time)
        # use processing times from last 200 frames
        if len(processing_times) > 200:
            processing_times.popleft()

        _, f_width = frame.shape[:2]
        # mean processing time [ms]
        processing_time = np.mean(processing_times) * 1000
        cv2.putText(
            frame,
            f"Inference time: {processing_time:.1f}ms",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            f_width / 1000,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"People Detected: {people_counter}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            f_width / 1000,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return frame, scores
