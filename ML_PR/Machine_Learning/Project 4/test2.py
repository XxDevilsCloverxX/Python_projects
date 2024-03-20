import numpy as np

def softmax(input_vector):
    # Calculate the exponent of each element in the input vector
    exponents = np.exp(input_vector)

    # Correct: divide the exponent of each value by the sum of the exponents
    # and round off to 3 decimal places
    sum_of_exponents = np.sum(exponents)
    probabilities = exponents / sum_of_exponents

    return probabilities

weights = np.random.uniform(low=0,high=1, size=(900, 2))

imgs = []
for n in range(100):
    img = np.random.randint(low=0,high=255, size=(30,30))
    img = img.flatten()
    img = img / 255
    imgs.append(img)

imgs = np.array(imgs)
labels = []
predictions = []
for img in imgs:
    pred = img@weights
    predictions.append(softmax(pred))
    labels.append(np.eye(weights.shape[1])[np.argmax(softmax(pred))])

predictions = np.array(predictions)
labels = np.array(labels)
print(np.hstack((predictions,labels)))