# data-generation-fish-swimming-model-based-naca-airfoil

This repository is to denerate fish swimming form image data and to create deep learning model for time series data for the Fishery industry. We tried to create fish swimming model for greater amberjack and detect fish sicks using the generated data.

# Preparation

First of all, you need to set up your Python environment which is version 3.11 to surely use this project. It might work well in other Python version environments.

And this project requires you to have google drive. Because the features data will be put on google drive and used by google colaboratory.

## Clone this repository

You need to clone this repository. Please clone this repository following the below command.

`git clone git@github.com:ko21suke/data-generation-fish-swimming-model-based-naca-airfoil.git`

## Install libraries to execute this project.

In this project, we use some Python libraries. So you need to install them. Please install those libraries following the below commands.

1. Move to your project directory.

    `cd ./data-generation-fish-swimming-model-based-naca-airfoil`

1. Install libraries.

    `pip install -r requirements.txt`

# Data generation

In this section, you will generate fish swimming sequential image data.  You can create sample images the following below.

## Prepare  CSV files

You need CSV files that have many parameters to define fish swimming form, swimming direction, and so on. We've prepared the sample CSV files in this repository to create sample data. The CSV files directory is `src/resource/input/params/`. If you want to use your CSV files, you need to set your files into the directory.

## Execute Python script to generate sequential images

When you generate fish swimming form images, you need to execute 'create_fish_swimming_image.py' following the below commands.

1. Move to the src directory.

    `cd ./src`

1. Execute Python script.

    `python create create_fish_swimming_image.py`

You will get many image data and a csv file.

  * Image data: `./resource/output/image/<created_datetime>/*.png`
    * The image data are created sequential fish swimming images. 
  * A csv file: `./resource/output/image/created_images_info.csv`
    * The file includes parameters you used and datetime.

# Deep learning model creation for time series data

In this section, you will build your machine learning model for fish swimming sequential data to detect fish sicks. The model is 
based on RNN model like Simple-RNN, GRU, and LSTM. You can chenge the model by changeing the model in `LSTM.ipynb` file.

## Analyze fish swimming form data

To create your machine learning model, you need to extract feature points. So you need to execute `calculate_feature_points_movement_distance.py `follwing the below command.

* `python calculate_feature_points_movement_distance.py`

You will get csv files into this directory `src/resource/output/csv/lstm`. The files have coordinates of feature points related to the fish's head, tail, center of the body and the movement distance between the previous image and the current image in the time series.

## Export CSV files to Google Drive

When you create your machine learning model, you will connect Google Drive to use feature data in `LSTM.ipynb`, So you need to Export the CSV files into your Google Drive.

1. Open Google Drive and Create new directory to import the CSV files.
1. Import lstm directory in this directory `src/resource/output/csv/` into created new direcotry in Google Drive. lstm directory has many CSV files.

## Execute all cells in the LSTM.ipynb

In this project, you use google colaboratory to create a machine learning model. So you need to open colabratory.

1. [Open google colaboratory](https://colab.research.google.com/)
1. Export this file `LSTM.ipynb` in this directory 'machine_learning/' to google colaboratory.

After you export `LSTM.ipynb` to your google colaboratory, you need to set your lstm directory path in your Google Drive to connect with your google colaboratory. What If your lstm directory in google drive is 'MyDrive/SampleDrive/ltsm', you need to set the string literal to `DATA_DIR` constant in 4th cell in LSTM.ipynb like `DATA_DIR = 'SampleDrive/lstm'`.

After you set `DATA_DIR`, you execute all cells to create your machine learning model.
Moreover, if you want to get another deep learning model for time series data, you rewrite the 11th cell like from `model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))` to ` model.add(GRU(128, input_shape=(X_train.shape[1], X_train.shape[2])))`.

The above example changes the model from LSTM to GRU.