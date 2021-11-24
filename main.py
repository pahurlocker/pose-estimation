import datetime

from flask import Flask, Response, render_template, request
from flask.json import jsonify
from flask_sqlalchemy import SQLAlchemy

from estimator import PoseEstimator

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data/database.db"
db = SQLAlchemy(app)


class PersonDetection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_name = db.Column(db.String(100), nullable=False)
    event_name = db.Column(db.String(50), nullable=False)
    create_date_time = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return "<PersonDetection %r>" % self.session_name


@app.route("/")
def index():
    return render_template("index.html")


def gen(estimator):
    people_counter = 0
    count = 0
    session_name = f"session-{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    while True:
        frame, scores = estimator.get_frame(jpeg_encoding=True)
        if count % 10 == 0:
            if len(scores) > people_counter:
                people_counter = len(scores)
                person = PersonDetection(
                    session_name=session_name, event_name="appeared"
                )
                db.session.add(person)
                db.session.commit()
            elif (len(scores) < people_counter) and (people_counter > 0):
                people_counter = len(scores)
                person = PersonDetection(
                    session_name=session_name, event_name="disappeared"
                )
                db.session.add(person)
                db.session.commit()
        count += 1

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/camera_feed")
def camera_feed():
    device_name = request.args.get("device_name", default="AUTO")
    precision = request.args.get("precision", default="FP16-INT8")
    return Response(
        gen(PoseEstimator(device_name=device_name, precision=precision)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/example_video")
def example_video():
    device_name = request.args.get("device_name", default="AUTO")
    precision = request.args.get("precision", default="FP16-INT8")
    return Response(
        gen(
            PoseEstimator(
                device_name=device_name,
                precision=precision,
                video_url="https://github.com/intel-iot-devkit/sample-videos/blob/master/store-aisle-detection.mp4?raw=true",
            )
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/detections")
def detections():
    page = request.args.get("page", default=1, type=int)
    per_page = request.args.get("per-page", default=100, type=int)
    detections = PersonDetection.query.paginate(page, per_page)
    results = {
        "results": [
            {
                "id": p.id,
                "session_name": p.session_name,
                "event_name": p.event_name,
                "create_date_time": p.create_date_time,
            }
            for p in detections.items
        ],
        "pagination": {
            "count": detections.total,
            "page": page,
            "per_page": per_page,
            "pages": detections.pages,
        },
    }
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
