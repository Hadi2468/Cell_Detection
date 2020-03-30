# Mask R-CNN for ONL Cell Detection

This is an implementation of Mask R-CNN model.

The repository includes:
* Logs directory, which shows the last trained waights for model
* Datasets directory, which consists of a small samples of our real train, validation, and test datasets
* Main code for training and testing in Jupyter notebook (Main_Code.ipynb, Main_Code_with_output.ipynb)
* Source code of Mask R-CNN (LICENSE, config.py, model.py, parallel_model.py, utils.py, visualize.py).
* Pre-trained weights for ImageNet (mask_RCNN.h5)

In this code I used some reference codes:
* Mask R-CNN model: https://github.com/matterport/Mask_RCNN
* COCO format: https://github.com/akTwelve/cocosynth

Also, I used the VIA 2.0.8 (VGG Image Annotator) software to annotate images, based on COCO standard format.
update