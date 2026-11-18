from flask import Flask, render_template, Response
import cv2
import time
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

app = Flask(__name__)

# 카메라 설정 (학생의 인덱스 0과 2 적용)
config_top = OpenCVCameraConfig(index_or_path=2, fps=30, width=640, height=480)
config_wrist = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)

top_cam = OpenCVCamera(config_top)
wrist_cam = OpenCVCamera(config_wrist)
top_cam.connect()
wrist_cam.connect()

def gen_frames(camera):
    while True:
        frame = camera.read()
        if frame is None: break
        # LeRobot의 RGB를 OpenCV의 BGR로 변환
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/top')
def video_top():
    return Response(gen_frames(top_cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/wrist')
def video_wrist():
    return Response(gen_frames(wrist_cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
      <body style="background: #222; color: white; text-align: center;">
        <h1>📸 Real-time Camera Stream</h1>
        <div style="display: flex; justify-content: center; gap: 20px;">
          <div><h3>Top View (Index 0)</h3><img src="/top" width="600"></div>
          <div><h3>Wrist View (Index 2)</h3><img src="/wrist" width="600"></div>
        </div>
      </body>
    </html>
    """

if __name__ == '__main__':
    print("🚀 웹 서버 시작! 브라우저에서 'http://localhost:5000'에 접속하세요.")
    app.run(host='0.0.0.0', port=5000, threaded=True)