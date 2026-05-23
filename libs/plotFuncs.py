import matplotlib.pyplot
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np

def plot_image(i, true_label, img, classesArray, predictions_array=[]):
    true_label, img = classesArray.index(true_label[i]), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    if len(predictions_array) > 0:
        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

    if len(predictions_array) > 0:
        plt.xlabel("{} {:2.0f}% ({})".format(classesArray[predicted_label],
                                    100*np.max(predictions_array),
                                    classesArray[true_label]),
                                    color=color)
    else:
        plt.xlabel("{}".format(classesArray[true_label]))

def plot_value_array(i, predictions_array, true_label, classesArray):
    true_label = classesArray.index(true_label[i])
    plt.grid(False)
    plt.xticks(range(len(predictions_array)), classesArray, rotation=90)
    plt.yticks([])
    thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
