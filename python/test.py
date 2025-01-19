import ModularCNN
import datasets
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

batch_size = 32
num_epochs = 10
save_dir = "models/train_0.bin"

ds = load_dataset("AlvaroVasquezAI/Animal_Image_Classification_Dataset")

dataset = ds["train"].train_test_split(test_size=0.1, seed=42)


def numpy_to_tensor(np_array, TensorClass):
    """
    Converts a NumPy array of shape (batch_size, channels, height, width)
    to a Tensor object using the exposed TensorClass from C++.

    Parameters:
        np_array (np.ndarray): The input NumPy array.
        TensorClass: The Tensor class from your pybind11 module.

    Returns:
        An instance of TensorClass containing the same data.
    """
    # Ensure the array is C-contiguous and of type float (or the type used by your Tensor)
    np_array = np.ascontiguousarray(np_array, dtype=np.float32)

    # Get the shape of the numpy array
    if np_array.ndim != 4:
        raise ValueError("Input array must have shape (batch_size, channels, height, width)")

    batch_size, channels, height, width = np_array.shape

    # Create a new Tensor object with the given dimensions.
    # The constructor takes: (batch_size, channels, height, width, value)
    # We set value=0.0 since we'll fill it from the numpy array.
    tensor_obj = TensorClass(batch_size, channels, height, width, 0.0)

    # Fill the tensor's internal data structure.
    # Assuming that tensor_obj.data is a nested structure: data[batch][channel][row][col].
    # This loop copies the data from the numpy array to the tensor.
    data = tensor_obj.data  # Exposed via pybind11
    for b in range(batch_size):
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    data[b][c][h][w] = float(np_array[b, c, h, w])

    return tensor_obj


def labels_to_tensor(labels_array, TensorClass):
    """
    Converts a NumPy array of labels to a Tensor object.

    Parameters:
        labels_array (np.ndarray): A 1D NumPy array of integer labels with shape (batch_size,).
        TensorClass: The Tensor class exposed from your pybind11 module.

    Returns:
        TensorClass: An instance of Tensor containing the labels.
    """
    # Ensure the input is a 1D NumPy array of integers
    if not isinstance(labels_array, np.ndarray):
        raise TypeError("labels_array must be a NumPy array")
    if labels_array.ndim != 1:
        raise ValueError("labels_array must be a 1D array of labels")
    if not issubclass(labels_array.dtype.type, np.integer):
        raise TypeError("labels_array must contain integer values")

    # Extract batch size
    batch_size = labels_array.shape[0]

    # Define the tensor dimensions for labels: (batch_size, 1, 1, 1)
    channels, height, width = 1, 1, 1

    # Create a Tensor object with the specified dimensions, initialized to 0.0
    tensor_labels = TensorClass(batch_size, channels, height, width, 0.0)

    # Access the tensor's data member
    data = tensor_labels.data  # Assuming 'data' is exposed via pybind11

    # Populate the tensor with label values
    for b in range(batch_size):
        data[b][0][0][0] = float(labels_array[b])  # Convert integer label to float

    return tensor_labels


def convert_image_to_array(example):
    image = example["image"]

    if image.mode != "RGB":
        image = image.convert("RGB")

    image_array = np.array(image)

    example["image"] = image_array
    return example


dataset.map(convert_image_to_array, batched=True)

train_ds = dataset["train"]
test_ds = dataset["test"]

optimizer = ModularCNN.AMSGrad(lr=1e-4, b1=0.965, b2=0.999, eps=1e-8, wd=1e-2)
criterion = ModularCNN.CrossEntropy()

layers = [ModularCNN.LayerConfig.conv(3, 16, 3, 3, 1, 1), ModularCNN.LayerConfig.pool(2, 2, 1, 1),
          ModularCNN.LayerConfig.conv(16, 32, 3, 3, 1, 1), ModularCNN.LayerConfig.pool(2, 2, 1, 1),
          ModularCNN.LayerConfig.conv(32, 64, 3, 3, 1, 1), ModularCNN.LayerConfig.pool(2, 2, 1, 1),
          ModularCNN.LayerConfig.fc(64 * 32 * 32, 128), ModularCNN.LayerConfig.fc(128, 3)]

model = ModularCNN.Model(layers)

for epoch in range(num_epochs):
    itera = tqdm(train_ds.iter(batch_size=batch_size))
    for batch in itera:
        images = batch["image"]
        labels = batch["label"]

        images = numpy_to_tensor(images, ModularCNN.Tensor)
        labels = labels_to_tensor(labels, ModularCNN.Tensor)

        predictions = model.forward(images)
        loss = criterion.forward(predictions, labels)
        grad = criterion.backward(predictions, labels)

        model.backward(grad)
        model.update(optimizer)
        model.zero_grad()

        itera.set_postfix(loss=loss)

    cuml_loss = 0
    for batch in tqdm(test_ds.iter(batch_size=batch_size)):
        images = batch["image"]
        labels = batch["label"]

        images = numpy_to_tensor(images, ModularCNN.Tensor)
        labels = labels_to_tensor(labels, ModularCNN.Tensor)

        predictions = model.forward(images)
        loss = criterion.forward(predictions, labels)

        cuml_loss += loss

        itera.set_postfix(loss=loss)

    print(f"Epoch {epoch + 1}, Eval Loss: {cuml_loss}")

model.saveWeights(save_dir)
