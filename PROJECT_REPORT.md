# Helmet Detection using YOLO

**Name:** Bilawal Khan 
**Course:** Data Science with AI  
**Date:** 12/10/2026

## 1. Abstract

In this project I built a model that can tell whether a motorcycle rider is wearing a helmet or not. I used a Kaggle dataset of 764 images with two classes, With Helmet and Without Helmet, and trained a YOLO26m model on it in Google Colab. After training I tested the model on images it had never seen. It reached a precision of [P], a recall of [R] and a mAP@50 of [mAP50] on the test set. I also made a small Gradio app so I could try the model on my own photos and videos. The biggest problem is that the dataset is small and has more helmet examples than no-helmet examples, so I do not think these results would stay the same on very different data.

## 2. Introduction

Riding a motorcycle without a helmet is one of the main reasons for serious head injuries in road accidents. Checking this by hand takes a lot of time and people cannot watch every road all day. I wanted to see if a computer vision model can do the first step, which is to find the heads in a picture and decide if each one has a helmet.

The goal of this assignment was to train a custom object detection model for a real world problem. I chose helmet detection and used YOLO because it is fast and easy to train with the Ultralytics library.

## 3. Dataset

I used a helmet detection dataset from Kaggle ([dataset name and link]). It comes as PNG images with Pascal VOC XML files for the boxes.

- 764 images in total
- 2 classes: With Helmet (962 boxes) and Without Helmet (489 boxes)
- There are about twice as many With Helmet boxes as Without Helmet boxes, so the classes are not balanced

Before training I checked the data for missing image or XML files, images that cannot be opened, empty annotation files and boxes that go outside the image. [Write here what the check in Cell 4 printed.] Boxes that went outside the image were clipped to the image border when I converted the labels.

I split the images randomly into 70% training, 20% validation and 10% test with a fixed seed (42), so the split is the same every time I run it. That gave me [train count] training, [val count] validation and [test count] test images (these numbers are printed in Cell 6).

## 4. Method

1. **Label conversion.** YOLO needs labels in its own text format, so I converted every Pascal VOC XML file into a YOLO label file (class 0 for With Helmet and class 1 for Without Helmet, with the box values scaled between 0 and 1).
2. **Data split.** I split the data 70/20/10 as described above and wrote a `data.yaml` file that tells YOLO where the images are.
3. **Training.** I fine-tuned the pretrained YOLO26m model in Google Colab on a T4 GPU. I used 100 epochs, image size 640, batch size 16 and seed 42. I also turned up the scale augmentation (`scale=0.9`), used a cosine learning rate (`cos_lr=True`), turned off mosaic augmentation for the last 15 epochs (`close_mosaic=15`) and set the class loss weight to 1.0 (`cls=1.0`).
4. **Early stopping.** I set patience to 30 and also wrote a small callback that stops training when the validation loss becomes higher than its best value so far. This saves time and helps to avoid overfitting.
5. **Evaluation.** I loaded the best weights and ran them on the test set to get precision, recall, mAP@50 and mAP@50-95.
6. **Testing on my own data.** I built a Gradio app where I can upload an image or a video and see the detections. I used a confidence of 0.25 and `agnostic_nms=True` so that one head does not get both labels at the same time.

## 5. Results

Results on the test set:

| Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|
| [P] | [R] | [mAP50] | [mAP50-95] |

- Training curves: [insert `results.png` here]
- Confusion matrix and precision-recall curve: [insert images from the `test_eval` folder here]
- Examples from my own images and video: [insert screenshots here]

## 6. Discussion

[Write 3 to 5 sentences about what you saw in your own results. For example: which class was harder for the model, what mistakes you noticed in the confusion matrix or in the example images, and what you think caused them.]

The test set is small, so a few wrong predictions can change the numbers a lot. For that reason I looked at the example images and not only at the metrics.

## 7. Limitations

- The dataset is small and has fewer Without Helmet examples, so the model can be weaker on that class.
- The test set is small, so the metrics are not very stable.
- For videos, the model looks at each frame separately and does not track people between frames, so the boxes can flicker.
- Blurry images, night scenes, people partly hidden and other things on the head (like caps) can cause wrong detections.
- The output is only a prediction from a model, so a person should check it before anyone takes action based on it.

## 8. Conclusion and Future Work

In this assignment I trained a YOLO26m model that detects With Helmet and Without Helmet heads and tested it on images and videos with a Gradio app. [Add one or two sentences about your actual result here.]

If I continue this project, I would add more images, especially for the Without Helmet class, include data from real traffic videos, try a larger image size, and add tracking so that each rider is counted only once in a video.

## 9. References

- Ultralytics YOLO documentation: https://docs.ultralytics.com
- Dataset: [Kaggle dataset name, author and link]
