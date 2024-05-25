import subprocess
import time

def run_dimensioner():
    subprocess.Popen(["python", "wrappers/python/examples/box_dimensioner_multicam/box_dimensioner_multicam_demo.py"])

def run_detection():
    subprocess.Popen(["python", "wrappers/tensorflow/example1-object_detection.py"])

if __name__ == "__main__":
    while True:
        run_dimensioner()
        time.sleep(5)  # 부피 측정 스크립트가 실행된 후 5초 동안 대기

        run_detection()
        time.sleep(5)  # 객체 탐지 스크립트가 실행된 후 5초 동안 대기
