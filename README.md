# KTH dataset multiaction recognition
## Webapp interface
![enter image description here](https://github.com/HitogamiAG/KTH-Multiaction-recognition/raw/master/images/webapp_screenshot_1.png)
![enter image description here](https://github.com/HitogamiAG/KTH-Multiaction-recognition/raw/master/images/webapp_screenshot_2.png)
## Install requirements
1. `python -m venv khnproject`
2. `source khnproject/bin/activate`
3. `pip install -r requirements.txt`
## Load your own data
1. Place videos into `data/` folder and distrubute them between train/val/test splits and class folders.
2. Run `python data_scripts.py ` with desired flags. List of available flags can be found in source code.
3. Python script generates csv files for all train/val/test splits
## Train & test model
1. When csv files are generated, run `python train.py` with desired flags to train model. List of available flags can be found in source code. Default save path is `checkpoints/`
2. When model is trained, run `python train.py` with desired flags to train model. List of available flags can be found in source code. You will get loss, accuracy, confusion matrix and classification report in the output of console.
## Run streamlit app
1. Observe `app.py` and optionally change path to the model weights
2. Run `streamlit run app.py` to run streamlit webapp. Put the link from the console into your web browser.
## Build & run docker container
1. Build your docker container using `docker build -t khn_app .`
2. After building is completed, run docker image using `docker run --rm -it -p 8889:8501 khn_app`
3. Open `localhost:8889` in your browser
4. Enjoy!

Pet project created by Alexandr Gavrilko. 09/01/2023.
