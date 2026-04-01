# Custom YOLO Models

Place custom head-detection models for the WebUI here.

- Linux / Windows: put `.pt` files in this folder
- macOS: put `.mlmodel` files in this folder

The WebUI settings page will list models found in this directory as custom YOLO options.

## Compatibility Note

Custom YOLO models should be prepared with the same Ultralytics major/minor environment used by this project.

- Linux / Windows environment file: `ultralytics==8.1.47`
- macOS environment file: `ultralytics==8.0.219`

Model compatibility may break if you export or train with a significantly different Ultralytics version.
