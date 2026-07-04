import cv2


def open_dual_camera(
    use_direct_camera: bool,
    top_index: int,
    wrist_index: int,
    top_stream_url: str,
    wrist_stream_url: str,
):
    """Open top/wrist camera from direct indices or network streams."""
    if use_direct_camera:
        cam_top = cv2.VideoCapture(top_index)
        cam_wrist = cv2.VideoCapture(wrist_index)
    else:
        cam_top = cv2.VideoCapture(top_stream_url)
        cam_wrist = cv2.VideoCapture(wrist_stream_url)

    for cam in (cam_top, cam_wrist):
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cam_top, cam_wrist

