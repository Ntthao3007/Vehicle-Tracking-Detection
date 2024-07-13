<<<<<<< HEAD
# AUTONOMOUS VEHICLE MICROSERVICE

The goal of this project is to develop an AI micro-system capable of detecting objects in video and predicting their motions in real time.

### Dataset ###

BDD100K [download here](https://www.vis.xyz/bdd100k/s)

### Run the inference pipeline ###
* ```python3 ./YOLO/Training/pipeline.py```

#### Training the model #####

To train the model use the script:
* ```python3 ./YOLO/Training/train.py ```

#### Interactive interface with gradio ####

* ```python3 ./YOLO/Training/gradio_app_image.py``` for image upload
* ```python3 ./YOLO/Training/gradio_app_video.pyy``` for video upload

### Fast API ###
* ```uvicorn main:app --host 0.0.0.0 --port 8000`` 

## Tools ## 
* ```pip install -r requirements.txt ``` 
=======
# Vehicle-Tracking-Detection
>>>>>>> 3294c5089f887eca6dbe3c9f5dde0f8f834f424b
