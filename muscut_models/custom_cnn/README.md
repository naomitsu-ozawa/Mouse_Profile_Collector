# Custom CNN Models

Place custom image-classification models for the WebUI here.

- Linux / Windows: put each SavedModel in its own subfolder under this directory
- Each model folder must contain `saved_model.pb`
- macOS: put `.mlmodel` files in this folder

The WebUI settings page will list models found in this directory as custom CNN options.

## Compatibility Note

Custom CNN models should be prepared for TensorFlow `2.15.x`.

- This project environment uses `tensorflow==2.15.1`
- TensorFlow `2.16` or later is not supported for these models

If you use a different TensorFlow version, model loading compatibility may break.
