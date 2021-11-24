import click
import cv2

from estimator import PoseEstimator


@click.command()
@click.option(
    "--device-name",
    help="device to run the network on, CPU, GPU, or MYRIAD. Default is CPU",
    default="CPU",
    type=str,
)
@click.option(
    "--precision",
    help="model precision FP16-INT8, FP16, or FP32. Default is FP16-INT8",
    default="FP16-INT8",
    type=str,
)
@click.option(
    "--video-url",
    help="use video instead of video camera, try https://github.com/intel-iot-devkit/sample-videos/blob/master/store-aisle-detection.mp4?raw=true. Default uses video camera",
    default="",
    type=str,
)
def main(device_name, precision, video_url):
    estimator = PoseEstimator(
        device_name=device_name, precision=precision, video_url=video_url
    )
    while True:
        frame, scores = estimator.get_frame()
        title = "Press ESC to Exit"
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
