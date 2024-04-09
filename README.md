# AMLSII_23-24_SN23187901

This project uses machine learning to solve the super resolution problems using DIV2K datasets from NTIRE 2017 Super-Resolution Challenge. This project will use downscaling factor = 4

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/penggang0719/AMLSII_23-24_SN23187901.git
    ```

2. Navigate to the project directory:
    ```
    cd 
    ```
    
3. Create a new conda environment from the `environment.yml` file:
    ```
    conda env create -f environment.yml
    ```

4. Activate the new environment:
    ```
    conda activate AMLS2
    ```

5. Install the required Python packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

This project trained two machine learning models, SRCNN and ESPCN. The trained models are saved in the folder A and B.

To get the test results, run the main.py.

The Results folder contains the randomly selected generated images using each model.

To understand and train the model step by step. Use train_step.ipynb to train and adjust the model. 

Training the model should take many time (more than 5 hours each)

## DIV2K Datasets reference

@InProceedings{Agustsson_2017_CVPR_Workshops,
	author = {Agustsson, Eirikur and Timofte, Radu},
	title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	month = {July},
	year = {2017}
} 

@InProceedings{Timofte_2017_CVPR_Workshops,
author = {Timofte, Radu and Agustsson, Eirikur and Van Gool, Luc and Yang, Ming-Hsuan and Zhang, Lei and Lim, Bee and others},
title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {July},
year = {2017}
}
