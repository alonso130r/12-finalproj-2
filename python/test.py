import ModularCNN
from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image
import numpy as np
from tqdm import tqdm

# Configuration
batch_size = 32
num_epochs = 10
save_dir = "models/train_0.bin"

# Load the full dataset
ds = load_dataset("AlvaroVasquezAI/Animal_Image_Classification_Dataset")

def convert_image_to_array(example):
    """
    Convert the PIL Image in the example to a NumPy array.
    If the image isn’t in RGB mode, convert it first.
    """
    try:
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_array = np.array(image)
        example["image"] = image_array
        return example
    except Exception as e:
        print(f"Error processing example: {e}")
        return None

def transform_dataset(dataset):
    """
    Process all examples in the dataset with a Python loop,
    applying the convert_image_to_array transformation.
    Returns a new Dataset with the transformed examples.
    """
    transformed_examples = []
    for idx in tqdm(range(len(dataset)), desc="Transforming dataset"):
        example = dataset[idx]
        new_example = convert_image_to_array(example)
        # Optionally: Skip any examples that cause an error.
        if new_example is not None:
            transformed_examples.append(new_example)

    if len(transformed_examples) == 0:
        raise ValueError("No transformed examples were produced.")

    # Reconstruct the dataset using the original features (if available)
    columns = transformed_examples[0].keys()
    data_dict = {column: [example[column] for example in transformed_examples]
                 for column in columns}
    new_dataset = Dataset.from_dict(data_dict, features=dataset.features)
    return new_dataset


# First, transform the entire dataset (un-split)
transformed_full_dataset = ds["train"]  # transform_dataset(ds["train"])

# Now perform a train/test split on the transformed dataset
split_ds = transformed_full_dataset.train_test_split(test_size=0.1, seed=24)

# Create a DatasetDict to hold both splits
dataset = DatasetDict({
    "train": split_ds["train"],
    "test": split_ds["test"]
})

print("Dataset preprocessed")
print("Initializing model")

# Set training and testing splits from our new dataset
train_ds = dataset["train"]
test_ds = dataset["test"]

# print(f"sample input dimensions: {train_ds[0]['image'].shape}")

train_ds.set_format(type="numpy", columns=["image", "label"])
test_ds.set_format(type="numpy", columns=["image", "label"])

# img = train_ds[-1]["image"]
# print(type(img))

# Define your custom tensor conversion functions
def numpy_to_tensor(np_array, TensorClass):
    """
    Converts a NumPy array of shape (batch_size, height, width, channels)
    to a Tensor object using the exposed TensorClass from C++.
    If the images are in HWC order, transpose them to CHW.
    """
    # If np_array.shape is (batch, height, width, channels), transpose to (batch, channels, height, width)
    if np_array.ndim != 4:
        raise ValueError("Input array must have 4 dimensions")

    # Check whether the last dimension is 3 or 1 (commonly the number of channels for an image)
    # and assume that if it's 3 (or 1) and not in the expected CHW order, we transpose it.
    batch_size, d2, d3, d4 = np_array.shape
    # A simple heuristic: if the last dimension is 3 or 1, and the second dimension is large (e.g., 256)
    # then the array is likely in HWC order.
    if d4 in (1, 3) and d2 > 10:
        np_array = np_array.transpose(0, 3, 1, 2)  # Rearranged to (batch, channels, height, width)

    np_array = np.ascontiguousarray(np_array, dtype=np.float32)
    batch_size, channels, height, width = np_array.shape

    tensor_obj = TensorClass(batch_size, channels, height, width, 0.0)
    data = tensor_obj.data  # Exposed via pybind11
    for b in range(batch_size):
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    data[b][c][h][w] = float(np_array[b, c, h, w])
    return tensor_obj



def labels_to_tensor(labels_array, TensorClass):
    """
    Converts a 1D NumPy array of integer labels to a Tensor object.
    """
    if not isinstance(labels_array, np.ndarray):
        raise TypeError("labels_array must be a NumPy array")
    if labels_array.ndim != 1:
        raise ValueError("labels_array must be a 1D array of labels")
    if not issubclass(labels_array.dtype.type, np.integer):
        raise TypeError("labels_array must contain integer values")
    batch_size = labels_array.shape[0]
    channels, height, width = 3, 1, 1
    tensor_labels = TensorClass(batch_size, channels, height, width, 0.0)
    data = tensor_labels.data
    for b in range(batch_size):
        data[b][0][0][0] = float(labels_array[b])
    return tensor_labels


# Initialize model, optimizer, criterion, and layer configurations
optimizer = ModularCNN.AMSGrad(1e-4, 0.965, 0.999, 1e-8, 1e-2)
criterion = ModularCNN.CrossEntropy(True)
layers = [
    ModularCNN.LayerConfig.conv(3, 4, 3, 3, 1, 1),
    ModularCNN.LayerConfig.pool(2, 2, 2, 0),
    ModularCNN.LayerConfig.conv(4, 8, 3, 3, 1, 1),
    ModularCNN.LayerConfig.pool(2, 2, 2, 0),
    ModularCNN.LayerConfig.conv(8, 16, 3, 3, 1, 1),
    ModularCNN.LayerConfig.pool(2, 2, 2, 0),
    ModularCNN.LayerConfig.fc(16384, 64),
    ModularCNN.LayerConfig.fc(64, 3)
]

model = ModularCNN.ModularCNN(layers)

print(f"Number of parameters: {model.getTotalParams()}")
print("Training model")

# Training and evaluation loops
for epoch in range(num_epochs):
    # Training loop
    train_iter = tqdm(train_ds.iter(batch_size=batch_size), desc=f"Epoch {epoch+1} (Train)")
    for batch in train_iter:
        images = batch["image"]
        labels = batch["label"]

        # labels = np.array(labels)

        # Convert NumPy images/labels to your custom Tensors
        images = numpy_to_tensor(images, ModularCNN.Tensor)
        labels = labels_to_tensor(labels, ModularCNN.Tensor)

        # print(len(images.data))
        # print(len(images.data[0]))
        # print(len(images.data[0][0]))
        # print(len(images.data[0][0][0]))

        predictions = model.forward(images)
        # print("Predictions")
        # print(len(predictions.data))
        # print(len(predictions.data[0]))
        # print(len(predictions.data[0][0]))
        # print(len(predictions.data[0][0][0]))
        # print("Labels")
        # print(len(labels.data))
        # print(len(labels.data[0]))
        # print(len(labels.data[0][0]))
        # print(len(labels.data[0][0][0]))
        loss = criterion.forward(predictions, labels)
        criterion.backward(predictions, labels)

        # print("Grad")
        # print(len(grad.data))
        # print(len(grad.data[0]))
        # print(len(grad.data[0][0]))
        # print(len(grad.data[0][0][0]))

        model.backward(predictions)
        model.update(optimizer)
        model.zeroGrad()

        train_iter.set_postfix(loss=loss)

    # Evaluation loop
    cuml_loss = 0
    test_iter = tqdm(test_ds.iter(batch_size=batch_size), desc=f"Epoch {epoch+1} (Test)")
    for batch in test_iter:
        images = batch["image"]
        labels = batch["label"]

        # labels = np.array(labels)

        images = numpy_to_tensor(images, ModularCNN.Tensor)
        labels = labels_to_tensor(labels, ModularCNN.Tensor)

        predictions = model.forward(images)
        loss = criterion.forward(predictions, labels)
        cuml_loss += loss

    print(f"Epoch {epoch+1}, Eval Loss: {cuml_loss}")

# Save the final model weights
model.saveWeights(save_dir)
