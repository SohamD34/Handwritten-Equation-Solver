# Handwritten-Equation-Solver


### Directory 
~~~
root
|--------- LICENSE
|--------- README.md
|--------- requirements.txt
|--------- .gitignore
|--------- my_dict.json
|--------- data
|           |------- character_data
|           |------- character_csv
|--------- test_data
|--------- utils
|           |------- utils.py
|           |------- extraction.py
|           |------- segmentation.py
|           |------- masked_cluster_segment.py
|--------- train.ipynb
|--------- adversarial_train.ipynb
|--------- equationsolver.ipynb
~~~

### Problem Statement

The primary objective of this research project is to develop a robust and accurate Handwritten Equation Solver using Deep Learning techniques.

Despite the significant strides made in the field of Optical Character Recognition (OCR) and Deep Learning, handwritten equation recognition and solving remains a complex task. This complexity arises from the intricate and complex nature of mathematical symbols, the diversity of handwriting styles, and the need for a nuanced understanding of mathematical symbols to accurately solve the equations.

Our project aims to address these challenges by proposing a relatively low-compute Deep Learning-based approach for handwritten equation recognition and solving. By leveraging image data of popular mathematical characters, we seek to develop a pipeline that can accurately recognize and solve handwritten mathematical equations. Our approach, by being relatively low-compute, aims to provide a viable alternative to the current SoTA solutions, particularly in resource-constrained environments.

* Setting up environment
~~~
conda create -n venv
~~~
* Activate the environment
~~~
conda activate venv
~~~
* In Git Bash, navigate to the desired folder location.
~~~
cd /desired_folder_path/
~~~
* Clone the repository.
~~~
git clone https://github.com/SohamD34/Handwritten-Equation-Solver.git
~~~
* Install requirements
~~~
pip install -r requirements.txt
~~~
* Download the dataset in the '/data' folder.
~~~
/root/Handwritten-Equation-Solver/data
~~~
* Segment the characters from the testing data
~~~
cd ../utils
python extraction.py
python masked_cluster_segment.py
~~~
* Run the training script using src/train.py. For adversarial training, use src/adversarial_train.py.
~~~
cd ../src
python train.py
python adversarial_train.py
~~~
* Run the solver script.
~~~
cd ..
jupyter notebook equationsolver.ipynb
~~~

### Dataset

The image dataset used for training the CharacterNet architecture was derived from the following sources.

- Dataset link: Handwritten Math Symbols [[Link](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols)]
- Custom created smaller version of dataset [[Link](https://drive.google.com/drive/folders/1kENoMUhq74NdzhTBcCX8PGKdBUrKGOb3?usp=sharing)]

Total data points = 16,500

Preprocessing of dataset:

- The images were resized to (128,128) shape for uniformity.
- All the images were converted into gray-scale (Black and White), 1 channel for uniformity.


### Results

Our architecture achieved accuracy of 95% on the training dataset and accuracy of 85% on the validation set for the simple CNN model, demonstrating its effectiveness in recognizing handwritten equations.
We performed adversarial training on the model using manipulated and noisy input images to make the model more robust, ensuring reliable performance in real-world scenarios in case of unseen and altered inputs.

![image](https://github.com/SohamD34/Handwritten-Equation-Solver/assets/96857578/d8196105-5167-4879-bc3b-0f4ef198494a)
![image](https://github.com/SohamD34/Handwritten-Equation-Solver/assets/96857578/d5c39741-0e21-4395-b760-32e223a9f954)


### Contributors  
- [Soham Niraj Deshmukh (B21EE067)](https://www.github.com/SohamD34)
- [Lakshit Choubisa (B21AI057)](https://github.com/Lakshit24sa)
- [Lokesh Tanwar (B21EE035)](https://github.com/Lokesh23102002)
- [Mohit Sharma (B21EE037)](https://github.com/mohitsharma-iitj)
