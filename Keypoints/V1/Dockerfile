FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04
RUN sudo apt-get update \
 && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 gcc \
 && sudo rm -rf /var/lib/apt/lists/*
RUN pip install opencv-python matplotlib albumentations numpy pycocotools
CMD ["python3", "-u", "/app/train.py"]
