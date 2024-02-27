import yaml
import subprocess
import math
import os
import numpy as np

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def print_config(config):
    for key, value in config.items():
        print(f"{key}: {value}")

########################################################
#                           get video duration
######################################################## 
def get_video_duration(video_path: str) -> float:
    """Get the duration of a video using ffprobe."""
    # Construct the command to get video duration
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-select_streams', 'v:0', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        video_path
    ]
    print(f"{video_path=}")
    try:
        # Execute the command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.stderr:
            print(f"Error getting video duration: {result.stderr}")
            return None
        # Strip whitespace and convert the output to float
        duration_str = result.stdout.strip()
        if duration_str:
            duration = float(duration_str)
            return duration
        else:
            print("No duration found in the ffprobe output.")
            return None
    except ValueError:
        print(f"Could not convert duration to float. Output was: '{result.stdout}'")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Failed to get video duration: {e}")
        return None

########################################################
#                           Split
######################################################## 
def split_video(video_path: str, output_folder: str, num_parts: int, total_duration: int) -> None:
    """
    Split a video into a fixed number of parts based on total duration and number of parts.

    Args:
    - video_path: Path to the input video file.
    - output_folder: Folder where the split parts will be saved.
    - num_parts: The number of parts to split the video into.
    - total_duration: The total duration of the video in seconds.
    """
    part_duration = total_duration / num_parts

    for part in range(num_parts):
        start_time = part * part_duration
        output_file_name = f"{output_folder}/part{part + 1}.mp4"

        # For the last part, adjust duration to cover the remainder of the video
        if part == num_parts - 1:
            command = [
                "ffmpeg", "-ss", str(start_time), "-i", video_path,
                "-c", "copy", "-to", str(total_duration), output_file_name, "-y"
            ]
        else:
            command = [
                "ffmpeg", "-ss", str(start_time), "-i", video_path,
                "-c", "copy", "-t", str(part_duration), output_file_name, "-y"
            ]

        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Part {part + 1} saved as {output_file_name}")

########################################################
#                           augmentation
######################################################## 
def get_augment_steps(config: dict, num_steps: int) -> dict:
   #E.g., augment_steps={'scale_factor': array([0.3,0.43333333,0.56666667,0.7]), 'blur_radius':array([....]),'saturation_factor': array([...])}
    # for scale_factor, blur_radius, saturation_factor,         num_steps=4
    augment_steps = {}

    params = config['augmentation_parameters']
    for param in params.keys():
        lower_border, upper_border = params[param]['low'], params[param]['high']
        augment_steps[param] = np.linspace(lower_border, upper_border, num_steps)

    return augment_steps

def augment_video(video_path: str, output_folder: str, augment_steps: dict, video_Idx: int) -> None:
    """
    Change the blur, saturation, and scale of a single .mp4 video file, saving the output
    to a folder named 'aug_parts' with '_aug' appended to the original file name before the extension.

    Args:
    - video_path: The path to the input video file.
    - blur_radius: The radius of the blur effect as a float.
    - saturation_factor: The factor by which to adjust the saturation level.
    - scale_factor: The scaling factor to apply to both width and height.
    """
    # Extract the directory and base name of the input video
    dir_name, base_name = os.path.split(video_path)
    name, ext = os.path.splitext(base_name)
       
    # Construct the output path by appending '_aug' before the extension and placing it in the aug_parts folder
    output_path = os.path.join(output_folder, f"{name}_aug{ext}")

    # get the corresponding parameters for the specific video
    blur_radius         = augment_steps['blur_radius'       ][video_Idx]
    saturation_factor   = augment_steps['saturation_factor' ][video_Idx]
    scale_factor        = augment_steps['scale_factor'      ][video_Idx]

    # Construct the filters
    blur_filter = f"boxblur={round(blur_radius)}:{round(blur_radius)}"
    saturation_filter = f"eq=saturation={saturation_factor}"
    scale_filter = f"scale=iw*{scale_factor}:ih*{scale_factor}"

    filters = f"{blur_filter},{saturation_filter},{scale_filter}"
    command = [
        'ffmpeg', '-i', video_path, '-filter_complex', filters, '-c:a', 'copy', output_path, '-y'
    ]
    
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if process.returncode != 0:
        print("Failed to modify video. Error:")
        print(process.stderr)
    else:
        print(f"Video adjusted with blur radius {blur_radius}, saturation factor {saturation_factor}, and scale factor {scale_factor}, saved in 'aug_parts' folder as {output_path}")




########################################################
#                           combine
######################################################## 
def combine_videos(input_folder: str, output_video_path: str) -> None:
    parts = [f for f in os.listdir(input_folder) if f.endswith('.mp4') and f.startswith('part')]
    parts.sort()

    if not parts:
        print("No video parts found in the specified folder.")
        return

    list_file_path = os.path.join(input_folder, "list.txt")
    with open(list_file_path, 'w') as list_file:
        for part in parts:
            list_file.write(f"file '{part}'\n")

    command = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file_path,
        "-c", "copy", output_video_path, "-y"
    ]

    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if process.returncode == 0:
        print(f"Combined video saved as {output_video_path}")
    else:
        print("Failed to combine video parts.")


########################################################
#                           main
######################################################## 

if __name__ == "__main__":
    config_file_path = 'config.yml'  # Replace with your config file path
    config = load_config(config_file_path)
    print_config(config)

    # Get duration of video and split into num_parts for different augmentation per each part
    duration = get_video_duration(config['input_file'])
    # Calculate the number of parts
    num_parts = math.ceil(duration / config['time_interval'])
    # calculate steps list for each blur(rounded)/scale/satruarion using np.linspace to get different scale per video parts (num_parts)
    augment_steps = get_augment_steps(config, num_steps=num_parts)  #E.g., {'scale_factor': array([0.3,...,0.7]), 'blur_radius':array([....]),'saturation_factor': array([...])}
    
    # Create split and augmented folder
    split_folder = 'video_parts' # only splitted video part
    aug_folder   = 'aug_parts'   # after augmentation to the parts
    os.makedirs(split_folder, exist_ok=True)
    os.makedirs(aug_folder  , exist_ok=True)

    # Split the video into parts and save them in temp_parts
    split_video(video_path=config['input_file'], output_folder=split_folder, num_parts=num_parts, total_duration=duration)

    # List all the video parts in temp_parts
    video_files = [f for f in os.listdir(split_folder) if f.endswith('.mp4')]

    # Augment the video parts and save them in aug_parts
    for video_Idx, video_file in enumerate(video_files):    
        input_path = os.path.join(split_folder, video_file)
        augment_video(video_path=input_path, output_folder=aug_folder, augment_steps=augment_steps, video_Idx=video_Idx)

    # Combine the parts back into a single video
    combine_videos(input_folder=aug_folder, output_video_path='output.mp4')
