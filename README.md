# AllerNet: Semantic Segmentation of Allergens in Food

Allernet, an Allergen semantic segmentation pipeline was the final project for 6.S058: Introduction to Computervision. The goal of this project was to create a novel and effective way to segment the big 9 dietary allergens. Utilizing a vision transformer, along with attension fusion and multi level attention, this pipeline segments allergens by predicting food labels and then converting those predictions to an allergen mask by mapping foods to their allergens. This project analyzed the Foodseg103 dataset and produced food segmentation results matching research papers. 

## Key Features

* **Novel Allergen Segmentation Pipeline:** The project proposes a unique approach to allergen segmentation by leveraging deep learning and existing food segmentation datasets.
* **BEITV2 Vision Transformer Architecture:** Utilizes the BEIT Transformer model, known for its efficiency and strong performance on segmentation tasks.
* **Attention Fusion:** Incorporates attention mechanisms to fuse information across different levels of the network, improving segmentation accuracy.
* **Two-Stage Approach:** The pipeline operates in two stages: first, predicting food item labels, and second, mapping those labels to their corresponding allergen masks.

***

## How it Works

The pipeline is built on the FoodSeg103 dataset, which contains 103 different food categories. AllerNet operates in two main phases:

1.  **Food Segmentation:** The pipeline first performs semantic segmentation to identify and label individual food items within an image.
2.  **Allergen Mapping:** Using a predefined mapping, the food labels are converted into allergen masks, highlighting the specific areas containing the "big 9" allergens.


***

## Getting Started

### Prerequisites

* Python 3.8+
* PyTorch
* TorchVision

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/samkrem/AllerNet.git
    cd allernet
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Prepare the dataset:** Download and place the FoodSeg103 dataset in the `data/` directory.
2.  **Train the model:**
    ```bash
    python train.py 
    ```
3.  **Evaluate Metrics:**
    ```bash
    python evaluate.py 
    ```

***

## Results

(Include a summary of your results here. You can mention key metrics like Mean IoU, Pixel Accuracy, etc. You could also include a table or a graph if you have the data.)

**Table 1: Food Prediction Performance Metrics on the Validation Set **

| Metric | Value |
| :--- | :--- |
| Mean IoU | 0.47 |
| Pixel Accuracy | 0.91 |
| Precision | 0.50 |
| Recall | 0.45 |

**Table 1: Allergen Prediction Performance Metrics on the Validation Set for Food**

| Metric | Value |
| :--- | :--- |
| Mean IoU | 0.48 |
| Pixel Accuracy | 0.92 |
| Precision | 0.52 |
| Recall | 0.46 |

***
