
## 1. Install Dependencies:


```bash
pip install -r requirements.txt
```

**2. Prepare Model Weights:**

You will need the YOLOv8 model weights or specification file. The default model used in the script is "yolov8n.yaml" it will download automaticaly. you can specify a different model using the `--model` argument.

**3. Prepare Test Video:**

You can specify the test video file using the `--testvideo` argument. The default test video file is "test.mkv". Make sure the video file exists in the specified path or provide a different path to your video.

**4. Choose a Mode:**

The script has two modes: "VISUAL_TEST" and "TRAIN". You can specify the mode using the `--mode` argument.

- `VISUAL_TEST`: This mode performs object detection on the specified test video. Detected objects will be highlighted in the video frames.
- `TRAIN`: This mode is for training the YOLOv8 model. You will need additional data and configurations for training.

**5. Run the Script:**

You can run the script with the desired mode and arguments. Here are a few examples:

- To perform a visual test with the default settings (use yolov8n.yaml and "test.mkv"):

```bash
python your_script_name.py
```

- To specify a different model and test video:

```bash
python your_script_name.py --mode VISUAL_TEST --model your_model.yaml --testvideo your_video.mp4
```

- To run the training mode (requires additional setup and data):

```bash
python your_script_name.py --mode TRAIN --model your_model.yaml
```

**6. Optional Saving:**

You can specify the `--save` argument as `True` if you want to save the result of visual test to a video file. The saved video will be named using the model and video file names.


