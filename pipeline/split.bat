@echo off
setlocal enabledelayedexpansion

rem Call the Python script to read the config.yml file
for /f "tokens=*" %%a in ('python read_config.py') do (
    if not defined input_file (
        set "input_file=%%a"
    ) else if not defined output_file (
        set "output_file=%%a"
    ) else if not defined time_interval (
        set "time_interval=%%a"
    )
)

rem Get input video duration and convert it to a number
for /f "delims=" %%a in ('ffprobe -v error -select_streams v:0 -show_entries format^=duration -of default^=noprint_wrappers^=1:nokey^=1 "%input_file%"') do set "duration=%%a"
for /f "delims=." %%a in ("%duration%") do set "duration_int=%%a"

rem Calculate the number of parts based on the total movie length and time interval
set /a "num_parts=!duration_int! / %time_interval%"
set /a "last_part_duration=!duration_int! %% %time_interval%"

rem If there is a remainder, add one more part to handle it
if !last_part_duration! gtr 0 (
    set /a "num_parts+=1"
)

rem Create a temporary directory to store the parts
if exist temp_parts rmdir /s /q temp_parts
mkdir temp_parts

rem Split the input video into parts
for /l %%i in (1, 1, !num_parts!) do (
    set /a "start_time=(%%i-1)*%time_interval%"
    set /a "end_time=%%i*%time_interval%"
    if %%i equ !num_parts! set "end_time=!duration_int!"
    
    echo Running ffmpeg for Part %%i...
    ffmpeg -i "%input_file%" -ss !start_time! -to !end_time! -c:v copy -c:a copy "temp_parts\part%%i.mp4"
    echo Finished processing Part %%i.
)

echo Script completed.
