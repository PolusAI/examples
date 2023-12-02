from __future__ import absolute_import, division, print_function
import os
import random
import streamlit as st
from streamlit_javascript import st_javascript
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow import keras
from keras import layers, Model
from keras.datasets import mnist

dashboard_folder = os.path.dirname(os.getenv("DASHBOARD_PATH"))
device_used = 'GPU' if len(tf.config.list_logical_devices('GPU'))>0 else 'CPU'
st_theme = st_javascript("""window.getComputedStyle(window.parent.document.getElementsByClassName("stApp")[0]).getPropertyValue("color-scheme")""")

# Below is markdown for Streamlit.
st.title('Neural Network Example')

st.subheader('Introduction')
    
st.columns(4)[1].image(dashboard_folder + '/pict/neural_network_overview.jfif', caption='', width=400)


st.markdown(
    """
        The dataset utilized in the below illustration is the MNIST dataset, comprising 60,000 examples for training and an additional 10,000 examples reserved for testing. Each of these images containing handwritten digits have been size-normalized and centered within a fixed-size image of 28x28 pixels, with the pixel values ranging from 0 to 255.

        In the process, every individual image undergoes a transformation, being converted to float32, and subsequently flattened into a one-dimensional array composed of 784 distinctive features—derived from the dimensions of the image (28x28).

    """, unsafe_allow_html=True)


st.columns(4)[1].image(dashboard_folder + '/pict/mnist_dataset_overview.png', caption='', width=400)


# MNIST dataset parameters.
num_classes = 10  # total classes (0-9 digits).
num_features = 784  # data features (img shape: 28*28).

# Default training parameters
training_steps = 500
display_step = 100
learning_rate = 0.001
batch_size = 256

# Form Streamlit Input

st.subheader('Training')
st.markdown(
"""
MNIST default dataset parameters.

**Number of classes** = 10 (total classes 0-9 digits).

**Number of features** = 784  (data features).
"""
)

# Initialize shuffle_button_state
shuffle_button_state = st.session_state.get('shuffle_button_state', False)

if "training" not in st.session_state:
    st.session_state.training = training_steps

if "display" not in st.session_state:
    st.session_state.display = display_step

if "learning" not in st.session_state:
    st.session_state.learning= learning_rate

if "batch" not in st.session_state:
    st.session_state.batch = batch_size

st.markdown('----')

placeholder1 = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()
placeholder4 = st.empty()

col1, col2, col3 = st.columns([6,1,1])

with col2:
    reset_button_state = st.button('Reset')
with col3:
    apply_button_state = st.button('Apply', type="primary")

st.markdown('----')

# Resetting the parameters
if reset_button_state:
    st.session_state.training = training_steps
    st.session_state.display = display_step
    st.session_state.learning = learning_rate
    st.session_state.batch = batch_size


tph = placeholder1.number_input(
    label='Training Steps', step=50, min_value=50, max_value=10000, key='training')
dph = placeholder2.number_input(
    label='Display Step', step=50, min_value=0, max_value=1000, key='display')
lph = placeholder3.number_input(
    label='Learning Rate', format="%.3f", min_value=0.000, max_value=2.000, step=0.001, key='learning')
bph = placeholder4.number_input(
    label='Batch Size', key='batch', step = 1, min_value=1, max_value=1000)

# Apply button
if not apply_button_state:
    if not shuffle_button_state:
        st.stop()


# Network parameters.
n_hidden_1 = 128  # 1st layer number of neurons.
n_hidden_2 = 256  # 2nd layer number of neurons.

# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape(
    [-1, num_features]), x_test.reshape([-1, num_features])

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# Create TF Model.
# NeuralNet class is based on Keras Model class
@keras.utils.register_keras_serializable()
class NeuralNet(Model):
    # Set layers.
    def __init__(self):
        super(NeuralNet, self).__init__()
        # First fully-connected hidden layer.
        self.fc1 = layers.Dense(n_hidden_1, activation=tf.nn.relu)
        # First fully-connected hidden layer.
        self.fc2 = layers.Dense(n_hidden_2, activation=tf.nn.relu)
        # Second fully-connecter hidden layer.
        self.out = layers.Dense(num_classes)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


# Build neural network model.
neural_net = NeuralNet()
# Cross-Entropy Loss.
# Note that this will apply 'softmax' to the logits.

def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

# Accuracy metric
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(
        tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.Adam(learning_rate)
# Optimization process.

def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = neural_net(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = neural_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))

@st.cache_resource(show_spinner=f"Training Neural Network on {device_used}...")
def train_model(training_steps, display_step, learning_rate, batch_size):
    step_list = []
    loss_list = []
    acc_list = []

    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
        # Run the optimization to update W and b values.
        run_optimization(batch_x, batch_y)

        if step % display_step == 0:
            pred = neural_net(batch_x, is_training=True)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accuracy(pred, batch_y)

            step_list.append(step)
            loss_list.append(loss)
            acc_list.append(acc)
    st.success('Done!')

    return step_list, loss_list, acc_list, neural_net

step_list, loss_list, acc_list, neural_net = train_model(tph, dph, lph, bph)



df = pd.DataFrame(
    data={"Step": step_list, "Loss": loss_list, "Accuracy": acc_list}
)

# CSS to inject contained in a string to hide index (first column) in Streamlit app.
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown (in Streamlit)
st.markdown(hide_table_row_index, unsafe_allow_html=True)

st.table(df)

# Test model on test set.
pred = neural_net(x_test, is_training=False)

# Below is markdown for Streamlit.
st.markdown(
    """
    Now testing model on test set.  

    """, unsafe_allow_html=True)

st.write("**_Test Accuracy:_** %f" % accuracy(pred, y_test))


st.subheader('Test')

# Below is markdown for Streamlit.
st.markdown(
    """
    **_Test image predictions are displayed below:_**
    """, unsafe_allow_html=True)

n_images = 5
test_images = x_test[:n_images]
#predictions = neural_net(test_images)

def shuffle_images():
    st.session_state.shuffle_button_state = True
    st.session_state.test_images = []
    st.session_state.test_labels = []
    for i in range(n_images):
        rand_index = random.randint(0, len(x_test)-1)
        st.session_state.test_images.append(x_test[rand_index])
        st.session_state.test_labels.append(y_test[rand_index])
    st.session_state.test_images = np.array(st.session_state.test_images)
    st.session_state.test_labels = np.array(st.session_state.test_labels)

if (not hasattr(st.session_state, 'test_images')):
    shuffle_images()

predictions = neural_net(st.session_state.test_images)

columns = st.columns(n_images+1)
columns[0].write("**Image:**")
columns[0].write("**Label:**")
columns[0].write("**Prediction:**")
for i, col in enumerate(columns[1:]):
    col.image(np.reshape(st.session_state.test_images[i], [28, 28]) / 255.)
    col.write(st.session_state.test_labels[i])
    col.write(np.argmax(predictions.numpy()[i]))


st.markdown(
    """


***If you want to reshuffle images, click the button below.***


    """
)

# Button Widget in Streamlit
st.button("Shuffle Images", on_click=shuffle_images)


st.subheader('Try It')

if st_theme == "dark":
    canvas_stroke_color="white",
    canvas_background_color="rgba(14, 17, 23, 1)",
    canvas_invert = False
else:
    canvas_stroke_color="black",
    canvas_background_color="white",
    canvas_invert = True

canvas = st_canvas(
    stroke_width=40, # defaults to 20
    stroke_color=canvas_stroke_color,
    background_color=canvas_background_color,
    update_streamlit=True,
    height=600,
    width =600,
    drawing_mode="freedraw",
    key="canvas",
)


if (
    canvas.json_data is not None
    and len(canvas.json_data["objects"]) != 0
):
    if 'canvas' in st.session_state:
        x = canvas
    # x is a numpy array 

    # Will only execute below logic (for predicting the sketch) if the button was clicked
    image = Image.fromarray(x.image_data)

    # Converting canvas with black stroke and white background to a canvas with white stroke and black background
    image = image.convert('RGB')
    if canvas_invert:
        image = ImageOps.invert(image)

    # Convert PIL image to to grayscale using 'L' mode
    grayscale_image = image.convert('L')

    # Resize the PIL image to 28x28 to fit neural model
    resized_image = grayscale_image.resize((28, 28))

    # Convert the resized PIL image back to a numpy array
    reshaped_image = np.array(resized_image)

    # Flatten the image to have a shape of (784,)
    flattened_image = reshaped_image.reshape(784)

    # Adding extra dimnesion to the input image
    input_data = flattened_image.reshape(-1, 1)

    input_data_final = np.squeeze(input_data)

    input_data_final_float = input_data_final.astype(np.float32)

    processed_image = Image.fromarray(input_data_final_float)

    drawn_image_tensor = tf.convert_to_tensor(input_data_final_float, dtype=tf.float32)
    drawn_image_tensor = tf.expand_dims(drawn_image_tensor, axis=0)


    prediction_drawable = neural_net(drawn_image_tensor, is_training=False)

    # Display model prediction on drawable image
    st.write(f"**Prediction:** {np.argmax(prediction_drawable.numpy())}")