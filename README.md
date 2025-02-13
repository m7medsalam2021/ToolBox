# Image Processing Toolbox

## Overview
This is a GUI-based image processing tool built using Python and Tkinter. It allows users to perform various transformations and enhancements on images, such as rotation, scaling, translation, skewing, flipping, contrast adjustments, and applying filters.

## Features
- **Add Image**: Load an image from the file system.
- **Reset Image**: Restore the original image after modifications.
- **Save Image**: Save the modified image to the file system.
- **Rotate Image**: Rotate the image between 0-360 degrees using a slider.
- **Scale Image**: Resize the image by specifying X and Y scale factors.
- **Translate Image**: Shift the image horizontally and vertically.
- **Skew Image**: Apply skew transformations to the image.
- **Flip Image**: Reflect the image horizontally or vertically.
- **Adjust Contrast**: Modify the image contrast using a slider.
- **Apply Filters**:
  - Histogram Equalization
  - Negative Transformation
  - Logarithmic Transformation
  - Power (Gamma) Correction
  - Bit Plane Slicing
  - Gray Level Slicing
- **Neighborhood Processing**:
  - Smoothing Spatial Filters (Low-pass filters, Pyramidal, Circular, Cone filters)
  - Median Filtering
  - Linear High-pass Filtering

## Requirements
- Python 3.x
- Required libraries:
  ```sh
  pip install numpy opencv-python pillow tkinter
  ```

## How to Run
1. Clone or download the project.
2. Install the required dependencies.
3. Run the script:
   ```sh
   python script_name.py
   ```

## Usage
1. Click "Add Image" to select an image file.
2. Apply transformations using the available controls.
3. Adjust contrast and apply filters as needed.
4. Save the processed image using the "Save" button.
5. Use the "Reset" button to restore the original image.

## Future Enhancements
- Add more advanced image processing techniques (edge detection, sharpening, etc.).
- Implement undo/redo functionality.
- Support batch processing of images.

## Credits
Developed using Python, OpenCV, and Tkinter.

