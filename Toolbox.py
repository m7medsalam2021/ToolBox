# importing libraries
import tkinter as tk
import cv2 
import numpy as np
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageEnhance

# Main 
class Main(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(' Tool Box')
        self.geometry('900x700')
       # variables for image
        self.original = None
        self.image = None

# Frame_1 20%
        frame1 = tk.Frame(self)
        frame1.place(relx=0, rely=0, relwidth=0.2, relheight=1)
# Frame_2 20%
        frame2 = tk.Frame(self)
        frame2.place(relx=0.2, rely=0, relwidth=0.2, relheight=1)
# upper frame
        upper_frame = tk.Frame(frame1)
        upper_frame.pack(fill='x', padx=8, pady=8, expand=True)
# middle frame
        middle_frame = tk.Frame(frame1)
        middle_frame.pack(fill='x', padx=8, pady=8, expand=True)
# bottom frame
        bottom_frame = tk.Frame(frame1)
        bottom_frame.pack(fill='x', padx=8, pady=8, expand=True)

# canvas
        self.canvas = tk.Canvas(self, background='grey')
        self.canvas.place(relx=0.4, rely=0, relwidth=0.7, relheight=1)

# Buttons
    # add button
        add_button = tk.Button(upper_frame, text='Add Image',command=self.add_img)
        add_button.pack(fill='x', padx=8, pady=8)
    # reset button
        reset_button = tk.Button(bottom_frame, text='Reset',command=self.reset_changes)
        reset_button.pack(fill='x', padx=8, pady=8)
    # save button
        save_button = tk.Button(bottom_frame, text='Save',command=self.save_image)
        save_button.pack(fill='x', padx=8, pady=8)

# Rotate
        rotate_frame= tk.Frame(middle_frame)
        rotate_frame.pack(fill='x', padx=8, pady=8, expand=True)
        
        rotate_frame.rowconfigure((0,1), weight=1, uniform='a')
        rotate_frame.columnconfigure((0,1), weight=1, uniform='a')
        
        rotate_label = tk.Label(rotate_frame, text='Rotate')
        rotate_label.grid(row=0, column=0, padx=8, pady=8, columnspan=2)
        
        rotate_slider = tk.Scale(rotate_frame, from_=0, to=360, orient=tk.HORIZONTAL, command = self.rotate_image)
        rotate_slider.grid(row=1, column=0, columnspan=2, sticky='we')
        
        
# Scale
        scale_frame= tk.Frame(middle_frame)
        scale_frame.pack(fill='x', padx=8, pady=8, expand=True)
        scale_frame.rowconfigure((0,1), weight=1, uniform='a')
        scale_frame.columnconfigure((0,1,2,3), weight=1, uniform='a')

        scale_button = tk.Button(scale_frame, text='Scale Image',command=self.scale_image)
        scale_button.grid(row=0, column=0, padx=8, pady=8, columnspan=4)

        scale_x_label = tk.Label(scale_frame,text='x')
        scale_x_label.grid(row=1, column=0, padx=8, pady=8)
        self.scale_x_entry = tk.Entry(scale_frame)
        self.scale_x_entry.grid(row=1, column=1, padx=8, pady=8)

        scale_y_label = tk.Label(scale_frame,text='y')
        scale_y_label.grid(row=1, column=2, padx=8, pady=8)
        self.scale_y_entry = tk.Entry(scale_frame)
        self.scale_y_entry.grid(row=1, column=3, padx=8, pady=8)
        

# Translation
        translation_frame= tk.Frame(middle_frame)
        translation_frame.pack(fill='x', padx=8, pady=8, expand=True)
        translation_frame.rowconfigure((0,1), weight=1, uniform='a')
        translation_frame.columnconfigure((0,1,2,3), weight=1, uniform='a')

        translation_button = tk.Button(translation_frame, text='Translate Image', command=self.translate_image)
        translation_button.grid(row=0, column=0, padx=8, pady=8, columnspan=4)

        translation_x_label = tk.Label(translation_frame,text='x')
        translation_x_label.grid(row=1, column=0, padx=8, pady=8)
        self.translation_x_entry = tk.Entry(translation_frame)
        self.translation_x_entry.grid(row=1, column=1, padx=8, pady=8)

        translation_y_label = tk.Label(translation_frame,text='y')
        translation_y_label.grid(row=1, column=2, padx=8, pady=8)
        self.translation_y_entry = tk.Entry(translation_frame)
        self.translation_y_entry.grid(row=1, column=3, padx=8, pady=8)
        
# Skewing
        skewing_frame = tk.Frame(middle_frame)
        skewing_frame.pack(fill='x', padx=8, pady=8, expand=True)
        skewing_frame.rowconfigure((0,1), weight=1, uniform='a')
        skewing_frame.columnconfigure((0,1,2,3), weight=1, uniform='a')

        skewing_button = tk.Button(skewing_frame, text='Skew', command=self.skew_image)
        skewing_button.grid(row=0, column=0, padx=8, pady=8, columnspan=4)

        skew_x_label = tk.Label(skewing_frame, text='X')
        skew_x_label.grid(row=1, column=0, padx=8, pady=8)
        self.skew_x_entry = tk.Entry(skewing_frame)
        self.skew_x_entry.grid(row=1, column=1, padx=8, pady=8)

        skew_y_label = tk.Label(skewing_frame, text='Y')
        skew_y_label.grid(row=1, column=2, padx=8, pady=8)
        self.skew_y_entry = tk.Entry(skewing_frame)
        self.skew_y_entry.grid(row=1, column=3, padx=8, pady=8)

    
# Reflect
        reflect_frame = tk.Frame(middle_frame)
        reflect_frame.pack(padx=10, pady=10)

        reflect_label = tk.Label(reflect_frame, text='Flip')
        reflect_label.pack(pady=5)
        reflect_horizontal_button = tk.Button(reflect_frame, text='Flip X', command=lambda: self.reflect_image('horizontal'))
        reflect_horizontal_button.pack(side='left', padx=5)

        reflect_vertical_button = tk.Button(reflect_frame, text='Flip Y', command=lambda: self.reflect_image('vertical'))
        reflect_vertical_button.pack(side='left', padx=5)
        

        
# Contrast
        contrast_frame= tk.Frame(frame2)
        contrast_frame.pack(fill='x', padx=8, pady=8, expand=True)
        
        contrast_frame.rowconfigure((0,1), weight=1, uniform='a')
        contrast_frame.columnconfigure((0,1), weight=1, uniform='a')
        
        contrast_label = tk.Label(contrast_frame, text='Contrast')
        contrast_label.grid(row=0, column=0, padx=8, pady=8, columnspan=2)
        
        contrast_slider = tk.Scale(contrast_frame, from_=50, to=150, orient=tk.HORIZONTAL, command=self.change_contrast)
        contrast_slider.grid(row=1, column=0, columnspan=2, sticky='we')
# Filters
        self.filters_label = tk.Label(frame2, text="Point Processing Filters")
        self.filters_combo = ttk.Combobox(frame2, state="readonly", values=["Histogram Equalization",
                                                                                 "Linear Transformation (Negative)",
                                                                                 "Logarithmic Transformations (ln x)",
                                                                                 "Power Transformation (Gamma Correction)",
                                                                                 "Bit Plane Slicing",
                                                                                 "Gray Level Slicing"])
        
        self.filters_combo.set("Select Filter")
        self.filters_combo.bind("<<ComboboxSelected>>")
        self.filters_combo.bind("<<ComboboxSelected>>", self.apply_filter)
        self.filters_label.pack(pady=5)
        self.filters_combo.pack(pady=5)

# Image Enhancement Neighborhood operations
        neighborhood_frame = tk.Frame(frame2)
        neighborhood_frame.pack(fill='x', padx=8, pady=8, expand=True)

        neighborhood_label = tk.Label(neighborhood_frame, text='Neighborhood processing')
        neighborhood_label.pack()

# Smoothing Spatial filters
        smoothing_frame = tk.Frame(neighborhood_frame)
        smoothing_frame.pack(fill='x', padx=8, pady=8, expand=True)
        smoothing_frame.rowconfigure((0, 1), weight=1, uniform='a')
        smoothing_frame.columnconfigure((0, 1, 2), weight=1, uniform='a')

        smoothing_label = tk.Label(smoothing_frame, text='Smoothing Spatial filters')
        smoothing_label.grid(row=0, column=0, padx=8, pady=8, columnspan=3)

        self.smoothing_combo = ttk.Combobox(smoothing_frame, state="readonly", values=["Linear Low pass Filter (traditional)",
                                                                                   "Pyramidal filter",
                                                                                   "Circular filter",
                                                                                   "Cone filter"])
        self.smoothing_combo.set("Select Smoothing Filter")
        self.smoothing_combo.bind("<<ComboboxSelected>>", self.apply_smoothing_filter)
        self.smoothing_combo.grid(row=1, column=0, padx=8, pady=8, columnspan=3)

# Median filter
        median_frame = tk.Frame(neighborhood_frame)
        median_frame.pack(fill='x', padx=8, pady=8, expand=True)

        median_button = tk.Button(median_frame, text='Median filter', command=self.median_filter)
        median_button.pack()

# Linear high pass filter
        high_pass_frame = tk.Frame(neighborhood_frame)
        high_pass_frame.pack(fill='x', padx=8, pady=8, expand=True)
        high_pass_frame.rowconfigure((0, 1), weight=1, uniform='a')
        high_pass_frame.columnconfigure((0, 1, 2), weight=1, uniform='a')

        high_pass_label = tk.Label(high_pass_frame, text='Linear high pass filter')
        high_pass_label.grid(row=0, column=0, padx=8, pady=8, columnspan=3)

        self.high_pass_combo = ttk.Combobox(high_pass_frame, state="readonly", values=["Sobel filter",
                                                                                   "Laplacian filter"])
        self.high_pass_combo.set("Select High Pass Filter")
        self.high_pass_combo.bind("<<ComboboxSelected>>", self.apply_high_pass_filter)
        self.high_pass_combo.grid(row=1, column=0, padx=8, pady=8, columnspan=3)

#Compression
        compression_button = tk.Button(frame2, text='Compress Image', command=self.compress_image)
        compression_button.pack(fill='x', padx=8, pady=8)


# Segmentation    
        segmentation_frame = tk.Frame(frame2)
        segmentation_frame.pack(fill='x', padx=8, pady=8, expand=True)

        segmentation_label = tk.Label(segmentation_frame, text='Image Segmentation')
        segmentation_label.pack()

        self.segmentation_combo = ttk.Combobox(segmentation_frame, state="readonly", values=["Thresholding",
                                                                                       "Edge-based Method",
                                                                                       "Gaussian",
                                                                                       "Laplacian"])
        self.segmentation_combo.set("Select Segmentation Method")
        self.segmentation_combo.pack(pady=5)
        self.segmentation_combo.bind("<<ComboboxSelected>>", self.apply_segmentation)
 # Sharpening        
        sharpening_frame = tk.Frame(frame2)
        sharpening_frame.pack(fill='x', padx=8, pady=8, expand=True)
        sharpening_frame.rowconfigure((0, 1), weight=1, uniform='a')
        sharpening_frame.columnconfigure((0, 1, 2), weight=1, uniform='a')

        sharpening_label = tk.Label(sharpening_frame, text='Sharpen Image')
        sharpening_label.grid(row=0, column=0, padx=8, pady=8, columnspan=3)

        sharpening_slider = tk.Scale(sharpening_frame, from_=0, to=5, orient=tk.HORIZONTAL, resolution=0.1,
                                     label="Sharpening Factor", command=self.sharpen_image)
        sharpening_slider.set(1.0)  # Default sharpening factor
        sharpening_slider.grid(row=1, column=0, columnspan=3, sticky='we')

# Functions
    def add_img(self):
        file_path = filedialog.askopenfilename(filetypes=[("Select An Image", "*.png;*.jpg;*.jpeg;*.tif")])
        if file_path:
            self.original = cv2.imread(file_path)
            self.image = self.original
            self.display_image()

        
 # Reset changes
    def reset_changes(self):
        self.image = self.original
        self.display_image()
    
# Save image
    def save_image(self):
        file_path = filedialog.asksaveasfilename(title="Save Image", defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            cv2.imwrite(file_path, self.image)

    def display_image(self):
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        tk_image = ImageTk.PhotoImage(Image.fromarray(rgb_image))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvas.tk_image = tk_image

    def sharpen_image(self, factor):
        if self.image is not None:
            sharpening_factor = float(factor)
            kernel = np.array([[-1, -1, -1],
                               [-1, 9 + sharpening_factor, -1],
                               [-1, -1, -1]])
            sharpened_image = cv2.filter2D(self.image, -1, kernel)
            self.image = sharpened_image
            self.display_image()  

    def rotate_image(self, value):
        center = (self.image.shape[1] // 2, self.image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, int(value), 1.0)
        self.image = cv2.warpAffine(self.original, rotation_matrix, (self.image.shape[1], self.image.shape[0]))
        self.display_image()

    def change_contrast(self, value):
        contrast_value = 1 + (float(value) - 50) / 50.0 # Adjust the scaling factor as needed
        pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced_image = enhancer.enhance(contrast_value)
        self.image = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
        self.display_image()

    def translate_image(self):
        tx = float(self.translation_x_entry.get())
        ty = float(self.translation_y_entry.get())
        TM = np.float32([[1, 0, tx], [0, 1, ty]])
        self.image = cv2.warpAffine(self.image, TM, (self.image.shape[1], self.image.shape[0]))
        self.display_image()
 
    def scale_image(self):
        x_str = self.scale_x_entry.get()
        y_str = self.scale_y_entry.get()
        if x_str != "":
            scale_x = float(self.scale_x_entry.get())
        else:
            scale_x = 1.0
        if y_str != "":
            scale_y = float(self.scale_y_entry.get())
        else:
            scale_y = 1.0
        
        self.image = cv2.resize(self.image, None, fx=scale_x, fy=scale_y)
        self.display_image()
        
    def skew_image(self):
        if self.image is not None:
            try:
                skew_x = float(self.skew_x_entry.get())
                skew_y = float(self.skew_y_entry.get())

                rows, cols = self.image.shape[:2]
                M = np.float32([[1, skew_x, 0], [skew_y, 1, 0]])

                self.image = cv2.warpAffine(self.image, M, (cols, rows))
                self.display_image()
            except ValueError:
                print("Please enter valid skew values.")
                
    def reflect_image(self, axis):
        if self.image is not None:
            if axis == 'horizontal':
                self.image = cv2.flip(self.image, 1)
            elif axis == 'vertical':
                self.image = cv2.flip(self.image, 0)
            self.display_image()

# Filter Functions
    def apply_filter(self, event):
        selected_filter = self.filters_combo.get()
        if selected_filter == "Histogram Equalization":
            self.histogram_equalization()
        elif selected_filter == "Linear Transformation (Negative)":
            self.linear_transformation_negative()
        elif selected_filter == "Logarithmic Transformations (ln x)":
            self.logarithmic_transform()
        elif selected_filter == "Power Transformation (Gamma Correction)":
            self.power_transformation()
        elif selected_filter == "Bit Plane Slicing":
            self.bit_plane_slicing()
        elif selected_filter == "Gray Level Slicing":
            self.gray_level_slicing()


    def histogram_equalization(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)
        self.image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
        self.display_image()

    def linear_transformation_negative(self):
        self.image = 255 - self.image
        self.display_image()
        
    def logarithmic_transform(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        logarithmic_image = np.log1p(gray_image)
        self.image = cv2.cvtColor(np.uint8(255 * (logarithmic_image / np.max(logarithmic_image))), cv2.COLOR_GRAY2BGR)
        self.display_image()
        
    def power_transformation(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gamma = 0.5  # Adjust gamma value as needed
        power_transformed_image = np.power(gray_image / 255.0, gamma) * 255
        self.image = cv2.cvtColor(np.uint8(power_transformed_image), cv2.COLOR_GRAY2BGR)
        self.display_image()
    
    def bit_plane_slicing(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        bit_plane = 7
        bit_sliced_image = (gray_image >> bit_plane) & 1
        bit_sliced_image *= 255
        self.image = cv2.cvtColor(np.uint8(bit_sliced_image), cv2.COLOR_GRAY2BGR)
        self.display_image()
    
    def gray_level_slicing(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        min_intensity = 100 
        max_intensity = 200 
        sliced_image = np.copy(gray_image)
        sliced_image[(gray_image >= min_intensity) & (gray_image <= max_intensity)] = 255
        sliced_image[(gray_image < min_intensity) | (gray_image > max_intensity)] = 0
        self.image = cv2.cvtColor(np.uint8(sliced_image), cv2.COLOR_GRAY2BGR)
        self.display_image()

    def median_filter(self):
            self.image = cv2.medianBlur(self.image, 5)  # Change the kernel size as needed
            self.display_image()
        
    def apply_smoothing_filter(self,event):
        selected_filter = self.smoothing_combo.get()
        if selected_filter == "Linear Low pass Filter (traditional)":
            self.linear_low_pass_filter()
        elif selected_filter == "Pyramidal filter":
            self.pyramidal_filter()
        elif selected_filter == "Circular filter":
            self.circular_filter()
        elif selected_filter == "Cone filter":
            self.cone_filter()

    def apply_high_pass_filter(self,event):
        selected_filter = self.high_pass_combo.get()
        if selected_filter == "Sobel filter":
            self.sobel_filter()
        elif selected_filter == "Laplacian filter":
            self.laplacian_filter()
    def linear_low_pass_filter(self):
            kernel = np.ones((3, 3), np.float32) / 9
            self.image = cv2.filter2D(self.image, -1, kernel)
            self.display_image()
        
    def pyramidal_filter(self):
        kernel = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], np.float32) / 10
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.display_image()
        
    def circular_filter(self):
        radius = 3
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        kernel[mask] = 1.0
        kernel /= kernel.sum()
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.display_image()

    def cone_filter(self):
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32) / 8
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.display_image()
        
    def sobel_filter(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_result = cv2.magnitude(sobel_x, sobel_y)
        self.image = cv2.cvtColor(np.uint8(sobel_result), cv2.COLOR_GRAY2BGR)
        self.display_image()
        

    def laplacian_filter(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        laplacian_result = cv2.Laplacian(gray_image, cv2.CV_64F)
        self.image = cv2.cvtColor(np.uint8(laplacian_result), cv2.COLOR_GRAY2BGR)
        self.display_image()
        
    
#Segmentation functions
            
    def apply_segmentation(self, event):
        selected_method = self.segmentation_combo.get()
        if selected_method == "Thresholding":
            self.thresholding_segmentation()
        elif selected_method == "Edge-based Method":
            self.edge_based_segmentation()
        elif selected_method == "Gaussian":
            self.gaussian_segmentation()
        elif selected_method == "Laplacian":
            self.laplacian_segmentation()

    def thresholding_segmentation(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        self.image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
        self.display_image()
        
    def edge_based_segmentation(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 50, 150)
        self.image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self.display_image()
    
    def gaussian_segmentation(self):
        self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
        self.display_image()
        

    def laplacian_segmentation(self):
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            laplacian_result = cv2.Laplacian(gray_image, cv2.CV_64F)
            self.image = cv2.cvtColor(np.uint8(laplacian_result), cv2.COLOR_GRAY2BGR)
            self.display_image()

#Compression
    def compress_image(self):
        if self.image is not None:
            quality = tk.simpledialog.askinteger("Compression", "Enter compression quality (1-100):", minvalue=1, maxvalue=100)
            if quality:
                file_path = filedialog.asksaveasfilename(title="Save Compressed Image", defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
                if file_path:

                    rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)

                    pil_image.save(file_path, "JPEG", quality=quality)
                    tk.messagebox.showinfo("Success", "Image compressed and saved successfully!")
        else:
            tk.messagebox.showwarning("Warning", "No image loaded to compress.")

            
if __name__ == '__main__':
    app = Main()
    app.mainloop()