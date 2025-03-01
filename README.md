# Traffic Sign Classification using Deep Learning (DL)

## Abstract

The proposed decision-making system utilizes image content characterization and a supervised classifier type of neural network. Image processing techniques for this kind of decision analysis involve preprocessing, feature extraction, and classification stages. During preprocessing, resize, color, and texture features are extracted from an input for training. The system will be used to classify the test images automatically to decide traffic sign characteristics.

## Scope of the Project

- **Data Analysis**
- **Data Preprocessing**
- **Training the Model**
- **Testing the Model**

## Working Model

In this model, the images are fed into a convolutional neural network (CNN) with a size of 30x30 and three color channels. The CNN consists of multiple convolutional layers with ReLU activation functions, followed by max-pooling layers and dropout for regularization. The output is then fed into fully connected layers, and the final layer uses softmax activation to predict 43 categories of traffic signs.

## Installation

Ensure you have Python installed (>=3.6). Install the required dependencies:

```sh
pip install numpy pandas matplotlib tensorflow pillow scikit-learn keras
```

## Usage

1. **Prepare Data**: Ensure you have the necessary dataset for training and testing.
2. **Load and Preprocess Data**: Load the data, resize, and normalize it.
3. **Build and Compile the Model**: Create a CNN model with convolutional, pooling, and dense layers. Compile the model with a loss function, optimizer, and metrics.
4. **Train the Model**: Train the model on the training data for a specified number of epochs.
5. **Evaluate the Model**: Evaluate the model on the testing data to check its performance.
6. **Make Predictions**: Use the trained model to make predictions on new data samples.

### Example Code

Here is an example of the complete code:

```python name=traffic_sign_classifier.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image
import os 
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Load Data
data = []
labels = []
classes = 43 
cur_path = os.getcwd() 
for i in range(classes): 
    path = os.path.join(cur_path, 'train', str(i)) 
    images = os.listdir(path) 
    for a in images:
        try:
            image = Image.open(path + '/' + a) 
            image = image.resize((30, 30)) 
            image = np.array(image) 
            data.append(image) 
            labels.append(i) 
        except: 
            print("Error loading image") 
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# CNN model building
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))
model.save("traffic_classifier.h5")

# Plotting graphs for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Plotting graphs for loss
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Evaluating the model on test data
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data = []
for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))
X_test = np.array(data)
pred = model.predict_classes(X_test)

# Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))
model.save('traffic_classifier.h5')
```

## Future Enhancements

- Improve model accuracy with more complex architectures.
- Add support for more traffic sign categories.
- Implement data augmentation for better generalization.
- Integrate visualization tools for model performance.

## Author

Developed by Pavan Kalyan Manda

Website Developer | IoT & Embedded Systems Enthusiast | AI/ML Developer
