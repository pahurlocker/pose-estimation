from flask import Flask, render_template, Response, request
from estimator import PoseEstimator

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def gen(estimator):
    while True:
        frame = estimator.get_frame(jpeg_encoding=True)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/video_feed")
def video_feed():
    device_name = request.args.get("device_name", default="CPU")
    return Response(
        gen(PoseEstimator(device_name=device_name)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
