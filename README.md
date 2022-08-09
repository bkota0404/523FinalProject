# Distracte Driver Detection 

The main objective of the project is to detect driver distractions using images which can be scaled into 4 wheeler dashcams and alert the drivers on go. It would help prevent several accidents on roads. The idea of the project was from a journal 'Distracted driver detection by combining in-vehicle and image data using deep learning' by Furkan Omerustaoglu, C. Okan Sakar and Gorkem Kar. The paper combined image data and vehicle data i.e., fuel consumption, torque etc to detect distracted drivers. We focused on image dataset and model optimization solely since neither the vehicle data nor the code is available publicly. We used VGG-16 and inception V3 models mentioned in the paper and along with used Resnet50 and AlexNet models to test the efficiency of these models against each other. The code also includes grad cam visualizations of randomly selected test data images along with ground truth labels and top 2 predictions. We measured the models efficiency using several metrics such as accuracy, precison, recall, f1-scores per class, top 2 accuracy. Resnet50 had highest accuracy with almost 91%.

## Dataset
The image dataset is publicly avaiable for training on Kaggle: https://www.kaggle.com/competitions/state-farm-distracted-driver-detection

As the test dataset labels are not publicly available, we used 10% of train dataset for testing. To partition the dataste, we tried randomly shuffling the dataset and picking equal proportions of images from the 10 classes used. But to measure the models performance more accurately, we decide to completely remove the images corresponding to two drivers from the training and use for testing. The train and test dataset can be found on the below link.
Training Data: https://drive.google.com/drive/folders/1cUG6jYCEPZNrHF4RGU4aA86XDk6ie5QF?usp=sharing
Test Data: https://drive.google.com/drive/folders/1W55URzNIsP61xEoK5HPtsDu4eOCanTWL?usp=sharing

