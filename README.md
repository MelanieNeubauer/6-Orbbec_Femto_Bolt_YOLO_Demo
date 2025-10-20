# RGB-D Data Collector with Orbbec Femto Bolt Camera

## About

This GUI-based tool allows you to capture synchronized RGB, depth, and segmentation mask images from an Orbbec Femto Bolt camera. It also automatically generates annotations in YOLO format, stores point clouds, and logs camera intrinsics and scene metadata.

### Camera Specifications 
Orbbec Femto Bolt

### Folder structure

```
├── main_basic.py                  # Entry point: GUI app for capturing & labeling
├── camera_interface.py      # RealSense camera setup and frame retrieval
├── segmentation_helper.py   # Depth segmentation + plane removal
├── annotation_writer.py     # YOLO-style annotation writer
├── requirements.txt         # Python dependencies
├── utils.py                 # Frame to BGR conversion
├── additional_info_code
├──── depth_info.py            # Script to convert .npy to depth image
├──── view_numpy.py            # Script to view .npy file as an image
├──── ply_viewer.py            # Script to view .ply images
├──── README.md                # This file
```

## Initial Setup procedure

1. User guide to install Orbbec SDK V2 Python Wrapper can also be found at [Orbbec SDK V2 Python Wrapper](https://orbbec.github.io/pyorbbecsdk/index.html)
    1. Open an empty project and clone the repository to get the latest version
    ```
    git clone   https://github.com/orbbec/pyorbbecsdk.git
    ```

    2. Install the necessary python development packages
    ```
    sudo apt-get install python3-dev python3-venv python3-pip python3-opencv
    ```
    3. Create a virtual environment and build the project
    ```
    python3 -m venv ./venv
    source venv/bin/activate
    cd pyorbbecsdk
    pip3 install -r requirements.txt
    cd pyorbbecsdk
    mkdir build
    cd build
    cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
    make -j4
    make install
    ```
    4. Set up the environment in pyorbbecsdk
    ```
    cd ..
    export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/
    sudo bash ./scripts/install_udev_rules.sh
    sudo udevadm control --reload-rules && sudo udevadm trigger
    ```
    5. Generate stubs for better IntelliSense support in your IDE
    ```
    source env.sh
    pip3 install pybind11-stubgen
    pybind11-stubgen pyorbbecsdk
    ```

2. Install the requirements
    ```
    cd ..
    pip install -r requirements.txt
    ```

3. Run the main.py file
    ````bash
    python3 main_yolo.py
    ````


## Run Code after Closing Window
Activate the virtual environment with pyorbbecsdk environment:
```
source .venv/bin/activate
source pyorbbecsdk/env.sh
```

Run your code:
```
python3 main.py
```