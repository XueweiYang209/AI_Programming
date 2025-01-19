from train import *

test_images = test_dataset.data.numpy().astype(np.float32)
test_labels = test_dataset.targets.numpy()
test_images_tensor = [TensorFull(image, requires_grad=False) for image in test_images]
test_labels_tensor = [TensorFull(label, requires_grad=False) for label in test_labels]
accuracy = model.predict(test_images_tensor, test_labels_tensor)
print(accuracy)