import cv2

def main(arguments):

    print(f'[python] Analyzing video')
    vid_path = arguments.get("vid_path")
    video = cv2.VideoCapture(vid_path)

    success, frame = video.read()
    while success:
        print(frame)
        success, frame = video.read()
    
    video.release()

    return True