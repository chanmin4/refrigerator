import os
import cv2
import boto3
import subprocess
import sys

# AWS 자격 증명 설정
#os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAZQ3DTF7OTIXRXZBL'
#os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
#os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

def get_kinesis_data_endpoint(stream_name, region_name='us-east-1'):
    client = boto3.client('kinesisvideo', region_name=region_name)
    response = client.get_data_endpoint(
        StreamName=stream_name,
        APIName='PUT_MEDIA'
    )
    return response['DataEndpoint']

def start_streaming(stream_name):
    endpoint = get_kinesis_data_endpoint(stream_name)
    region_name = 'us-east-1'
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    # GStreamer 명령어 생성
    gst_str = (
        f"appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        f"x264enc bitrate=5000000 ! "
        f"h264parse ! kvssink stream-name={stream_name} "
        f"storage-size=512 aws-region={region_name} "
        f"access-key='AKIAZQ3DTF7OTIXRXZBL'"
        f"secret-key='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'"
        f"endpoint={endpoint}"
    )

    out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30.0, (640, 480), True)
    
    if not out.isOpened():
        print("Failed to open GStreamer pipeline.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        out.write(frame)

        # ESC 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()

if __name__ == "__main__":
    stream_name = 'lifecam'
    start_streaming(stream_name)
