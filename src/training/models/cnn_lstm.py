
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2


def build_cnn_lstm(
    input_shape=(16, 224, 224, 3),
    lstm_units=128,
    dropout_rate=0.5
):

    # Video Input
    video_input = layers.Input(shape=input_shape)

    # Pretrained CNN Backbone
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze CNN initially
    base_model.trainable = False

    # Extract features from every frame
    x = layers.TimeDistributed(base_model)(video_input)

    # Convert feature maps into feature vectors
    x = layers.TimeDistributed(
        layers.GlobalAveragePooling2D()
    )(x)

    # Learn temporal relationships
    x = layers.LSTM(lstm_units)(x)

    # Regularization
    x = layers.Dropout(dropout_rate)(x)

    # Binary Classification
    output = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = models.Model(
        inputs=video_input,
        outputs=output,
        name="Fight_Detection_CNN_LSTM"
    )

    return model


def build_cnn_lstm_fine_tune(
    input_shape=(16, 224, 224, 3),
    lstm_units=128,
    dropout_rate=0.5,
    fine_tune=False
):

    video_input = layers.Input(shape=input_shape)

    # Load pretrained CNN
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze or unfreeze CNN
    base_model.trainable = fine_tune

    if fine_tune:
        # Freeze early layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Keep Batch Normalization frozen
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    # Extract frame features
    x = layers.TimeDistributed(base_model,name="frame_feature_extractor")(video_input)

    # Spatial feature pooling
    x = layers.TimeDistributed(
        layers.GlobalAveragePooling2D()
    )(x)

    # Temporal feature learning
    x = layers.LSTM(lstm_units)(x)

    # Regularization
    x = layers.Dropout(dropout_rate)(x)

    # Binary classification
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(
        inputs=video_input,
        outputs=output,
        name="Fight_Detection_CNN_LSTM"
    )

    return model
