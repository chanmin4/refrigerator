import subprocess
import sys

# 현재 환경에서 설치된 패키지들을 확인하고 리스트에 저장
installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().split('\n')
installed_packages = [package.split('==')[0] for package in installed_packages]

# requirements.txt 파일에서 이미 설치된 패키지들을 제외하고 나머지 패키지들을 리스트에 저장
with open('requirements.txt', 'r') as file:
    required_packages = file.read().split('\n')
required_packages = [package for package in required_packages if package not in installed_packages]

# requirements-exclude-installed.txt 파일에 나머지 패키지들을 저장
with open('requirements-exclude-installed.txt', 'w') as file:
    file.write('\n'.join(required_packages))
