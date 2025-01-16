# import ModularCNN
import datasets
from datasets import load_dataset

ds = load_dataset("AlvaroVasquezAI/Animal_Image_Classification_Dataset")

# Let's do an 80/10/10 split
split_dataset = ds["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]


print(train_dataset, test_dataset)
