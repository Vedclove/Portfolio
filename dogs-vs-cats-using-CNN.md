---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: base
  language: python
  name: python3
---

# Dogs/Cats Classification using CNN

```{code-cell} ipython3
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
```

## Load Dataset

```{code-cell} ipython3
#Training Algorithm
X_train = np.loadtxt('input.csv', delimiter = ',')
Y_train = np.loadtxt('labels.csv', delimiter = ',')

#Testing Data
X_test = np.loadtxt('input_test.csv', delimiter = ',')
Y_test = np.loadtxt('labels_test.csv', delimiter = ',')
```

```{code-cell} ipython3
# As per the requirement the images are reshaped and converted
X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)

X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)

X_train = X_train/255.0
X_test = X_test/255.0
```

```{code-cell} ipython3
print("Shape of X_train: ", X_train.shape)
print("Shape of Y_train: ", Y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_test: ", Y_test.shape)
```

```{code-cell} ipython3
# Random number is generated that used as an index to point towards a specific image in the data set
idx = random.randint(0, len(X_train))
plt.imshow(X_train[idx, :])
plt.show()
```

## Model

```{code-cell} ipython3
model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (100, 100, 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(32, (3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(64, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])
```

```{code-cell} ipython3
model = Sequential()

model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (100, 100, 3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
```

```{code-cell} ipython3
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
```

```{code-cell} ipython3
model.fit(X_train, Y_train, epochs = 5, batch_size = 64)
```

```{code-cell} ipython3
model.evaluate(X_test, Y_test)
```

## Making predictions

```{code-cell} ipython3
# Image URL (replace with your URL)
image_url = 'https://as2.ftcdn.net/jpg/00/97/58/97/1000_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.webp'

# Fetch the image from the URL
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
#img = Image.open('dogscheck.png')
# Resize the image to 100x100 pixels (assuming your model expects 100x100 input)
img = img.resize((100, 100))

# Convert image to a numpy array and normalize it
img_array = np.array(img) / 255.0  # Normalize pixel values between 0 and 1

# Make sure the image has 3 color channels (RGB)
if img_array.shape[-1] != 3:
    img_array = np.stack([img_array] * 3, axis=-1)  # Convert grayscale to RGB if needed

# Display the image
plt.imshow(img)
plt.show()

# Predict using the model (reshape to the expected input shape of your model)
y_pred = model.predict(img_array.reshape(1, 100, 100, 3))

# Convert output to binary (if necessary)
y_pred = y_pred > 0.5

# Map the prediction
if y_pred == 0:
    pred = 'Dog'
else:
    pred = 'Billi hai andha hai kya'

print("It is a:", pred)
```
