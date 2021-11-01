import cv2
import click
from estimator import PoseEstimator


@click.command()
@click.option(
    "--device-name",
    help="device to run the network on, CPU or MYRIAD. Default is CPU",
    default="CPU",
    type=str,
)
def main(device_name):
    estimator = PoseEstimator(device_name=device_name)
    while True:
        frame = estimator.get_frame()
        title = "Press ESC to Exit"
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title, frame)
        key = cv2.waitKey(1)
        # escape = 27
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
