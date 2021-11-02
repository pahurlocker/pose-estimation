import cv2
import collections
import time

import numpy as np
from numpy.lib.stride_tricks import as_strided
from openvino import inference_engine as ie

from decoder import OpenPoseDecoder
from player import VideoPlayer

model_path = f"models/human-pose-estimation-0001.xml"
model_weights_path = f"models/human-pose-estimation-0001.bin"

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
    def __init__(self, device_name):
        # initialize inference engine
        ie_core = ie.IECore()
        # read the network and corresponding weights from file
        net = ie_core.read_network(model=model_path, weights=model_weights_path)
        # load the model on the CPU (you can use GPU or MYRIAD as well)
        self.exec_net = ie_core.load_network(net, device_name)

        # get input and output names of nodes
        self.input_key = list(self.exec_net.input_info)[0]
        self.output_keys = list(self.exec_net.outputs.keys())

        # get input size
        self.height, self.width = self.exec_net.input_info[
            self.input_key
        ].tensor_desc.dims[2:]
        self.player = VideoPlayer(source=0, flip=False, fps=30, skip_first_frames=0)
        self.decoder = OpenPoseDecoder()

        # start capturing
        self.player.start()

    def __del__(self):
        self.player.stop()

    def get_frame(self, jpeg_encoding=False):
        # jpeg = self._run_pose_estimation()
        # return jpeg.tobytes()
        frame = self._run_pose_estimation()
        if jpeg_encoding:
            _, encoded_img = cv2.imencode(
                    ".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90]
                )
            return encoded_img.tobytes()
        else:
            return frame


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
    def _process_results(self, img, results):
        pafs = results[self.output_keys[0]]
        heatmaps = results[self.output_keys[1]]

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
        output_shape = self.exec_net.outputs[self.output_keys[0]].shape
        output_scale = img.shape[1] / output_shape[3], img.shape[0] / output_shape[2]
        # multiply coordinates by scaling factor
        poses[:, :, :2] *= output_scale

        return poses, scores

    def _draw_poses(self, img, poses, point_score_threshold, skeleton=default_skeleton):
        if poses.size == 0:
            return img

        img_limbs = np.copy(img)
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            # Draw joints.
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > point_score_threshold:
                    cv2.circle(img, tuple(p), 1, colors[i], 2)
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
        cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
        return img

    def _run_pose_estimation(self):
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
        results = self.exec_net.infer(inputs={self.input_key: input_img})
        stop_time = time.time()
        # get poses from network results
        poses, scores = self._process_results(frame, results)

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
            1,
            cv2.LINE_AA,
        )

        return frame