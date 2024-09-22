
## Vision-based landing algorithm for autonomous drone

An algorithm of object detection and localization for motion planning of an autonomous drone.
To provide optimum safety autonomously, a vision based safe landing 
algorithm is designed. the drone will be capable of detecting any 
objects that may block it and plans motion in 2D plane to avoid them during the 
landing phase. This problem will be solved by implementing yoloV5 tesnorRT
model for object detection and creating an occupancy grid map to 
decide which is the best place to land on. motion commands will be sent over 
MAVlink protocol to the flight controller to drive the drone.

![photo_2024-09-22_18-03-03](https://github.com/user-attachments/assets/0b90ae47-b7ca-41bf-bd3b-89b1d1b34f03)
![photo_2024-09-22_18-02-54](https://github.com/user-attachments/assets/1651eca6-977e-433b-95f3-01f364516a7c)
![photo_2024-09-22_18-02-49](https://github.com/user-attachments/assets/f167f15a-1f64-4aeb-b1cf-c4140920ad7b)

https://github.com/user-attachments/assets/6c15184e-2f29-4b12-804a-312abc212346

## Project flight video
you can watch the full video of the project in the link below.
https://www.youtube.com/watch?v=jyFPdx2TSI4




## Running YoloV5 with TensorRT Engine on Jetson.
==========

This is a step by step guide to build and convert YoloV5 model into a TensorRT engine on Jetson. This has been tested on Jetson Nano.

Please install Jetpack OS version 4.6 as mentioned by Nvidia and follow below steps. Please follow each steps exactly mentioned :

Jetson Nano:


Install Libraries
=============
Please install below libraries::

    $ sudo apt-get update
	$ sudo apt-get install -y liblapack-dev libblas-dev gfortran libfreetype6-dev libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev
	$ sudo apt-get install -y python3-pip
	

Install below python packages
=============
Numpy comes pre installed with Jetpack, so make sure you uninstall it first and then confirm if it's uninstalled or not. Then install below packages:

    $ numpy==1.19.0
	$ pandas==0.22.0
	$ Pillow==8.4.0
	$ PyYAML==3.12
	$ scipy==1.5.4
	$ psutil
	$ tqdm==4.64.1
	$ imutils

Install PyCuda
=============
We need to first export few paths

	$ export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
	$ export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
	$ python3 -m pip install pycuda --user
	

Install Seaborn
=============

    $ sudo apt install python3-seaborn
	
Install torch & torchvision
=============

	$ wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
	$ pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
	$ git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
	$ cd torchvision
	$ sudo python3 setup.py install 
	
### Not required but good library
sudo python3 -m pip install -U jetson-stats==3.1.4

This marks the installation of all the required libraries.

------------------------------------------------------------------------------------------

Generate wts file from pt file
=============
Yolov5s.pt and Yolov5n.pt are already provided in the repo. But if you want you can download any other version of the yolov5 model. Then run below command to convert .pt file into .wts file 

	$ cd JetsonYoloV5
	$ python3 gen_wts.py -w yolov5s.pt -o yolov5s.wts
	
Make
=============
Create a build directory inside yolov5. Copy and paste generated wts file into build directory and run below commands. If using custom model, make sure to update kNumClas in yolov5/src/config.h

	$ cd yolov5/
	$ mkdir build
	$ cd build
	$ cp ../../yolov5s.wts .
	$ cmake ..
	$ make 
	
Build Engine file 
=============

    $ ./yolov5_det -s yolov5s.wts yolov5s.engine s
	

Testing Engine file 
=============

	$ ./yolov5_det -d yolov5s.engine ../images
	
This will do inferencing over images and output will be saved in build directory.

-----------------------------------------------------------------------------------------

Python Object Detection
=============
Use `app.py` to do inferencing on any video file or camera.

	$ python3 app.py

If you have custom model, make sure to update categories as per your classes in `yolovDet.py` .


## Credits
for more of jetson nano set up guides please refer to
- https://github.com/mailrocketsystems

