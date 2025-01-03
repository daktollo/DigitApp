import tensorflow as tf
from tensorflow.keras import models, layers


class MnistModelTrainer:
    def load_dataset(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def create_model(self):
        self.model = models.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        

    def train_model(self, epochs=5):
        self.model.fit(self.x_train, self.y_train, epochs=epochs)

    def save_model(self, model_name):
        self.model.save(model_name)


if __name__ == "__main__":
    train_model = MnistModelTrainer()
    train_model.load_dataset()
    train_model.create_model()
    train_model.train_model(epochs=5)
    train_model.save_model("mnist_model.h5")
