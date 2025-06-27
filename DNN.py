from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow import keras
import matplotlib.pyplot as plt
from atlasify import atlasify
import scienceplots
from sklearn.metrics import roc_curve, auc

plt.style.use(["science", "no-latex"])

def training_model(training, validation, testing):
    # training, validation and testing are the x_train and y_train combined respectively.
    # trains model using datasets with engineered features
    x_train = training[:, 0:-1]
    y_train = training[:, -1]
    x_val = validation[:, 0:-1]
    y_val = validation[:, -1]
    x_test = testing[:, 0:-1]
    y_test = testing[:, -1]
    print(x_train)
    print(y_train)

    epochs = 100000

    # callbacks
    history = History()
    stopping = EarlyStopping(monitor="val_accuracy", patience=100)

    # model
    model = Sequential()
    model.add(Dense(20, input_shape=(41,), activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs, batch_size=1000, validation_data=(x_val, y_val), callbacks=[history, stopping])

    # accuracy metrics, writes to a file some specifics about the model and performance for comparison.
    # if file doesnt exist, create it
    _, accuracy = model.evaluate(x_test, y_test)
    f = open("Model_Accuracy.txt", "a")
    f.write("[20]-0.001-1000" + " Accuracy: " + str(accuracy) + "\n")
    f.close()

    # save model
    model.save("[20]-0.001-1000.keras")

    # data for plots
    acc = np.array(history.history["accuracy"])
    loss = np.array(history.history["loss"])
    val_acc = np.array(history.history["val_accuracy"])
    val_loss = np.array(history.history["val_loss"])
    step = np.arange(len(acc))

    f1 = plt.figure(1)
    plt.plot(step, loss, label="training loss")
    plt.plot(step, val_loss, label="validation loss")
    atlasify("Simulation Work in Progress")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.title("Loss over Epochs")

    f2 = plt.figure(2)
    plt.plot(step, loss-val_loss)
    atlasify("Simulation Work in Progress")
    plt.ylabel("delta loss")
    plt.xlabel("epochs")
    plt.title("Training - Validation Loss")

    f3 = plt.figure(3)
    plt.plot(step, acc, label="training accuracy")
    plt.plot(step, val_acc, label="validation accuracy")
    atlasify("Simulation Work in Progress")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.title("Accuracy over Epochs")

    f4 = plt.figure(4)
    plt.plot(step, acc - val_acc)
    atlasify("Simulation Work in Progress")
    plt.ylabel("delta accuracy")
    plt.xlabel("epochs")
    plt.title("Training - Validation Accuracy")

    plt.show()



def performance(testing):
    # ROC and AUC of the model
    x_test = testing[:, 0:-1]
    y_test = testing[:, -1]

    model = load_model("[20]-0.001-1000.keras")

    predictions = model.predict(x_test)



    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    model_auc = auc(fpr, tpr)

    f1 = plt.figure(1, figsize=(10,10))
    plt.plot(fpr, tpr, label = "AUC = " + str(model_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend()

    f2 = plt.figure(2, figsize=(10,10))
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr, tpr, label="AUC = " + str(model_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve Zoomed Into Top Left Corner')
    plt.legend()

    plt.show()