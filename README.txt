RETINA AI - PHASE 2 (5-layer CNN + 4 pre-trained models)
========================================================

This program looks at a retina (eye) photo and detects diabetic
retinopathy (DR):  No DR  /  DR

Phase 1 used PCA + Logistic Regression and was about 70% accurate.
Phase 2 replaces it with a Convolutional Neural Network (CNN) with
5 convolutional layers, trained from scratch. The data
pre-processing (preprocess.py) is the same as in Phase 1.

Models 2-5 are 4 models pre-trained on ImageNet (transfer learning):
ResNet-18, EfficientNet-B0, DenseNet-121 and MobileNetV3-Large.
They share one set of files (pretrained.py, train_pretrained.py,
predict_pretrained.py) and are tested on the same photos as the CNN,
so all 5 models can be compared.

THE CNN  (cnn.py)
-----------------
  input: pre-processed photo, 48 x 48 x 3 (from preprocess.py)
  conv layer 1 :  32 filters 3x3 + BatchNorm + ReLU            -> 48 x 48
  conv layer 2 :  64 filters 3x3 + BatchNorm + ReLU + MaxPool  -> 24 x 24
  conv layer 3 : 128 filters 3x3 + BatchNorm + ReLU + MaxPool  -> 12 x 12
  conv layer 4 : 256 filters 3x3 + BatchNorm + ReLU + MaxPool  ->  6 x 6
  conv layer 5 : 512 filters 3x3 + BatchNorm + ReLU + MaxPool  ->  3 x 3
  global average pooling -> dropout -> dense output layer (No DR / DR)
  about 1.57 million weights

HOW TO RUN  (terminal)
----------------------
The APTOS 2019 data (Kaggle) must be in the folder "dataset":
    dataset/train.csv
    dataset/train_images/*.png

Once, inside this folder:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Then:
    python3 preprocess.py     Step 1: photos -> dataset/X.npy, dataset/y.npy
    python3 train_cnn.py      Step 2: trains the CNN -> cnn_dr.pt, outputs/dr/
    python3 predict.py samples/grade2_moderate.png     Step 3: test one photo

train_cnn.py uses the Apple GPU (MPS) automatically. It prints one line
per epoch and the TEST RESULTS at the end.
5 grades: python3 train_cnn.py --task grade   (No DR / Mild / Moderate /
Severe / Proliferative, saved as cnn_grade.pt and outputs/grade/;
predict.py then shows the grade too)

The "samples" folder has one example photo for each grade; the
file name tells you the doctor's real grade, so you can compare.

HOW THE CNN IS TRAINED  (train_cnn.py)
--------------------------------------
* Split: 70% train / 15% validation / 15% test, stratified by grade.
  Duplicate photos in APTOS (the same eye saved twice) are kept in the
  same split, so the CNN is never tested on a photo it trained on.
* Data augmentation: in every epoch each training photo is randomly
  turned by 90 degrees, mirrored, shifted a few pixels and changed in
  brightness / contrast.
* AdamW optimiser with a one-cycle learning rate, 40 epochs, batch
  size 32. Dropout, weight decay and label smoothing against overfitting.
* The epoch with the best validation accuracy is kept.
* The accuracy is measured once, on test photos the CNN has never
  seen. A prediction averages the photo and 3 mirrored copies of it.

MODELS 2-5: 4 PRE-TRAINED MODELS  (pretrained.py)
-------------------------------------------------
Each model first learned to see on ImageNet (1.2 million everyday photos).
Its 1000-class output layer is replaced by a new dense layer for our
classes and ALL layers are fine-tuned on the retina photos.
  input: the same 48 x 48 photo, enlarged to 128 x 128, BGR -> RGB
  pre-trained feature layers -> global average pooling -> dropout -> new dense output layer

  model               main idea                                         weights   download
  ResNet-18           residual blocks (skip connections)                11.2 M    45 MB
  EfficientNet-B0     MBConv + squeeze-and-excitation, compound scaling  4.0 M    21 MB
  DenseNet-121        dense blocks (each layer sees all earlier ones)    7.0 M    31 MB
  MobileNetV3-Large   light inverted-residual blocks, hard-swish         3.0 M    21 MB

Train (one command per model and task; --model is one of
resnet18 | efficientnet_b0 | densenet121 | mobilenet_v3_large):
    python3 train_pretrained.py --model efficientnet_b0 --task dr      No DR / DR
    python3 train_pretrained.py --model efficientnet_b0 --task grade   5 grades
  -> efficientnet_b0_dr.pt / efficientnet_b0_grade.pt and outputs/efficientnet_b0_<task>/

All 4 models, both tasks, one after another:
    for m in resnet18 efficientnet_b0 densenet121 mobilenet_v3_large; do
      for t in dr grade; do python3 train_pretrained.py --model $m --task $t; done
    done

Test one photo with all 4:
    python3 predict_pretrained.py samples/grade2_moderate.png

Same split, augmentation and test photos as the CNN; 10 epochs, one-cycle
learning rate (peak 0.0005 for ResNet-18 / DenseNet-121, 0.001 for
EfficientNet-B0 / MobileNetV3), AdamW. Options: --epochs, --lr, --batch-size.
The first run of each model downloads its ImageNet weights (needs internet).
(train_resnet.py and predict_resnet.py still work as shortcuts for ResNet-18.)

WEBSITE  (server.py + frontend/ - the one that is deployed on Render)
--------------------------------------------------------------------
Pages: Home (about the project), Screening (upload a photo -> all 10
models, votes, probabilities), Training results (comparison charts,
validation curves, all metrics, confusion matrices) and Technical
(data, pre-processing, training setup, architectures, deployment).
The server runs the models as ONNX with onnxruntime - no PyTorch - so all
10 models fit in ~280 MB of RAM (Render's free plan has 512 MB).

Step 1 - export the trained models (again after every new training):
    pip install onnx onnxscript onnxconverter-common
    python3 export_web.py         -> web_models/*.onnx + web_models/manifest.json

Step 2 - build the website (needs Node.js 20+; skip if frontend/out exists
and you did not change the frontend):
    cd frontend && npm install && npm run build && cd ..

Step 3 - run it on this computer:
    pip install -r requirements-web.txt     (once)
    python3 server.py             then open http://127.0.0.1:8000

Editing the frontend with live reload: run python3 server.py in one
terminal and  cd frontend && npm run dev  in another, then open
http://localhost:3000 .

DEPLOY ON RENDER (free)
-----------------------
1. Push this folder to a GitHub repository (.gitignore leaves out the
   dataset, the .pt files and venv; web_models/ (~110 MB) IS pushed):
       git init && git add . && git commit -m "RetinaAI website"
       git branch -M main
       git remote add origin https://github.com/<you>/<repo>.git
       git push -u origin main
2. render.com -> New + -> Blueprint -> pick the repository (render.yaml is
   found automatically) -> Apply.
   (Or: New + -> Web Service -> the repository -> Language "Docker" ->
   Instance type "Free" -> Health check path /api/health.)
3. The first build takes ~5-10 min. The site is then at
   https://<name>.onrender.com . Every git push redeploys it.
Free services sleep after 15 min without visitors; the first visit after
that takes ~1 min to wake up.

Old simple page (PyTorch, this computer only): python3 app.py

OUTPUTS  (outputs/dr/ after training)
-------------------------------------
training_log.txt       everything printed during training, incl. the test accuracy
metrics.json           accuracy, sensitivity, specificity, precision, ROC-AUC
training_curves.png    loss and accuracy per epoch (training vs validation)
confusion_matrix.png   confusion matrix on the test photos

EXAMPLE OUTPUT  (predict.py)
----------------------------
Image      : samples/grade2_moderate.png

DR DETECTION  (5-layer CNN, ...% accurate on unseen test images)
Prediction : DR
Probabilities:
  No DR         :  ...%
  DR            :  ...%

FILES
-----
cnn.py            the 5-layer CNN
preprocess.py     image cleaning (same as Phase 1)
train_cnn.py      trains and tests the CNN, saves outputs/
predict.py        checks one image
pretrained.py         Models 2-5: the 4 pre-trained models (ImageNet)
train_pretrained.py   trains and tests one of them, saves <model>_<task>.pt and outputs/<model>_<task>/
predict_pretrained.py checks one image with all 4
resnet.py             ResNet-18 part (used by pretrained.py)
train_resnet.py       shortcut: train_pretrained.py --model resnet18
predict_resnet.py     checks one image with ResNet-18 only
server.py         the website server (ONNX, deployed on Render)
frontend/         the website (Next.js): Home, Screening, Training results, Technical
export_web.py     .pt -> web_models/*.onnx + manifest.json for the website
web_models/       exported models used by the website
Dockerfile, render.yaml, requirements-web.txt   deployment on Render
app.py            old simple local page (web/index.html), needs PyTorch
requirements.txt  library versions
samples/          5 example retina photos (grades 0-4)
dataset/          APTOS data (train.csv, train_images/) + X.npy, y.npy
cnn_dr.pt         the trained CNN (created by train_cnn.py)
<model>_<task>.pt the trained pre-trained models, e.g. resnet18_dr.pt,
                  efficientnet_b0_grade.pt (created by train_pretrained.py)
outputs/          training logs, metrics and plots (dr/, grade/ = CNN,
                  <model>_dr/, <model>_grade/ = pre-trained models)

NOTES
-----
* The 90%+ target is for DR detection (No DR / DR). Telling the exact
  grade (5 classes) is much harder - even a pre-trained EfficientNet-B0
  reaches only about 86% in our full RetinaAI project - so
  --task grade will score lower.
* It is a student project, not a medical tool.
