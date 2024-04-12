# Handwritten-Equation-Solver


*Team Members*  
- Lakshit Choubisa (B21AI057)
- Lokesh Tanwar (B21EE035)
- Mohit Sharma (B21EE037)
- Soham Niraj Deshmukh (B21EE067)

### Contents

| Sr. No. | Title                     | Pg.No. |
|---------|---------------------------|--------|
| 1.      | Problem Statement         | 3      |
| 2.      | Solution Strategy         | 4      |
| 3.      | Dataset                   | 6      |
| 4.      | Methodology and Innovations | 7    |
| 5.      | Results                   | 9      |
| 6.      | Critical Analysis         | 11     |
| 7.      | Conclusion                | 12     |
| 8.      | References                | 13     |

### Problem Statement

The primary objective of this research paper is to develop a robust and accurate Handwritten Equation Solver using Deep Learning techniques.

Despite the significant strides made in the field of Optical Character Recognition (OCR) and Deep Learning, handwritten equation recognition and solving remains a complex task. This complexity arises from the intricate and complex nature of mathematical symbols, the diversity of handwriting styles, and the need for a nuanced understanding of mathematical symbols to accurately solve the equations.

Our project aims to address these challenges by proposing a relatively low-compute Deep Learning-based approach for handwritten equation recognition and solving. By leveraging image data of popular mathematical characters, we seek to develop a pipeline that can accurately recognize and solve handwritten mathematical equations. Our approach, by being relatively low-compute, aims to provide a viable alternative to the current SoTA solutions, particularly in resource-constrained environments.

### Dataset

The image dataset used for training the CharacterNet architecture was derived from the following sources.

- Dataset link: Handwritten Math Symbols [https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols]
- Custom created smaller version of dataset [https://drive.google.com/drive/folders/1kENoMUhq74NdzhTBcCX8PGKdBUrKGOb3?usp=sharing]

Total data points = 16,500

Preprocessing of dataset:

- The images were resized to (128,128) shape for uniformity.
- All the images were converted into gray-scale (Black and White), 1 channel for uniformity.


### Results

Our architecture achieved accuracy of 95% on the training dataset and accuracy of 85% on the validation set for the simple CNN model, demonstrating its effectiveness in recognizing handwritten equations.
We performed adversarial training on the model using manipulated and noisy input images to make the model more robust, ensuring reliable performance in real-world scenarios in case of unseen and altered inputs.
