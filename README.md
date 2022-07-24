# Projeto de MLOps: Road Traffic Accidents Classification with Neural Network

**PROGRAMA DE PÓS-GRADUAÇÃO EM ENGENHARIA ELÉTRICA E DE COMPUTAÇÃO (PPgEEC) - UFRN**

**Disciplina:** Aprendizagem de Máquina **(EEC1509)**

**Grupo:** 

*   Beatriz Soares de Souza **(20211020152)**
*   Pedro Victor Andrade Alves **(20211027716)**

**Dataset:** [Road_Traffic_Accidents.csv](https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents)

![Accident](https://lh3.googleusercontent.com/pw/AM-JKLX-DKD3hwss7RQ3TLRC_oSKvfWMLhhHUMbxTJWDe1PzV21gBNnqM_FYzzrRCiI8FhHjEc2UhXIKvLIK2xfGaEsxEo2Gna6wfbygyNULPBglcEt8wwNKM0Nt_y-t3TMMdnWKXJgkq7vbnlzU9HkW2gL8=w500-h320-no?authuser=0)

## 1 - Project's overview:

### 1.1 -Chosen Dataset:

The dataset [Road_Traffic_Accidents.csv](https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents), from Kaggle, was collected from Addis Ababa Sub city police departments for Masters research work and it has been prepared from manual records of road traffic accident of the year 2017-2020. All the sensitive information have been excluded during data encoding and finally it has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms.

The datasets consists of the following features:

- Age_band_of_driver
- Sex_of_driver
- Educational_level
- Vehicle_driver_relation
- Driving_experience
- Lanes_or_Medians
- Types_of_Junction
- Road_surface_type
- Light_conditions
- Weather_conditions
- Type_of_collision
- Vehicle_movement
- Pedestrian_movement
- Cause_of_accident
- Accident_severity

### 1.2 - Workflow and Deploy:

In this project you will find the steps necessary for building a classification model, such as setting up the environment, performing exploratory data analysis (EDA), processing, training and testing your model, and deploying a pipeline in production. All the code used on the project can be also found in this [Jupyter Notebook](https://github.com/pedro048/Neural_Network_Road-Traffic-Accidents_MLOps/blob/main/Project_NN_Road_Traffic_Accidents/Project_NN_Road_Traffic_Accidents.ipynb), since we're gonna be using Python. The main goal here is to predict the severity of a road traffic accidents, based on variables available in the dataset, deploy it to Heroku and test it live, as [a Deliveried API ](). Neural network was the classification method used and severity of road traffic accidents can reach three diffrent levels: 0, 1, 2.


## 2 - Workflow:

The pipeline in charge of initiate the process of place a ML model in production can be found in the image below:

![Workflow](https://lh3.googleusercontent.com/pw/AM-JKLUCw27d6nW0YYrq-zIshMFLMSbCssGxQtNiQMwzxGu7W83kIgmfWlg75IKaNCCDuIB2Dk2ZTGLfEyvDt-AsW3F9m_MIVMlJoBPomkgBolc3WuSYvM2E3uFNDtcFgwhNE-dj1EcEMTtkhi8qmqBzvv9H=w1496-h948-no?authuser=0)

The steps illustrated in the picture above are:

### **2.1 - Extract, Transform and Load (ETL)**

Before we start creating our model we need to import the dataset, send the raw data to a tracking tool for machine learning (in our case, that will be Wandb), summarize our dataset using pandas profiling, and “clean” the data such as removing the duplicates.

### **2.2 -  Data Check**

With our preprocessed data ready, it’s important to run some tests to make sure that we have enough data to continue, and that all columns follow their own classes pattern.

### **2.3 - Data Segregation**

To train our model, we need to split train and test data first, then we further divide the dataset into train and validation.

### **2.4 - Train**

In this step we will prepare our data: remove the outliers, try to minimize the cardinality of some categorical features. Then, finally, we can define the hyperparameters of the pipeline (hyperparameter tuning) to actually train our model and find the best configuration through wandb sweep. **Among the options to improve learning, we have chosen**

  - Configure capacity with node and layers
  - Configure gradient precision with batch size
  - Configure what to optimize with loss function
  - Fix vanishing gradient with relu
  - Accelerate learning with batch normalization
  
**Applied techniques to improve generalization**

  - Decouple layers with dropout
  - Halt training at the right time with early stopping


### **2.5 - Test**

To test the model, we are going to use the test/validation data and run it in the model.

### **2.6 - Project's workflow:** https://github.com/pedro048/Neural_Network_Road-Traffic-Accidents_MLOps/blob/main/Project_NN_Road_Traffic_Accidents/Project_NN_Road_Traffic_Accidents.ipynb 

## 3 - Deploy:

All the process described in the image below is able to deliver a functional API:

![Deploy](https://lh3.googleusercontent.com/pw/AM-JKLXlXm09RhjF_hoHXWz4MhCFU9jF4VKlJr1OIcSinD0itYVclc8fJdRqTA6ECoBKcX7QLT8Ln8tRukszIyxMZIr_Y75nUxSp9DtY-xYXFHyVoDy6fSXCY_lU2mgjAqqdGL4lYrFOwsC22Eh9Tx55JgHz=w1598-h949-no?authuser=0)

We deployed the model using the FastAPI package and incorporated it into a CI/CD framework using GitHub Actions. It is important to understand that FastAPI is a modern API framework that allows us to write code quickly while maintaining flexibility. Having our API tested, we deployed it to Heroku and tested it live.

You can find the implemented API in the source/api/main.py whereas tests are on source/api/test_main.py.

### **3.1 - API Delivered with Heroku:** 

## **4 - Article on Medium**:

## **5 - Comparing the results with a Decision Tree model**:

### **5.1 - Neural Network**

![NN_Metric](https://lh3.googleusercontent.com/pw/AM-JKLX4qyNjIcznycOeqAkHjAca9ZY5RzxM5j93NV_huCIWDhIAW-uBvEG9sKDtQU8gRX3CPibBT6Ub_4ZeHs9fdhOYoiQz9JDgE-FxpMAP3jUriq2Zo3NokRhR_T4OApdXxikQNo3CLHvk7_8MWJIUJJyu=w1805-h948-no?authuser=0)

### **5.2 - Decision Tree**

![DT_Metric](https://lh3.googleusercontent.com/pw/AM-JKLW6KKIS-QqDNPc1rXQKDgykKNzcIcwovdvhPeEjnymGjPh5EGvLx3iRxNvnYPxjQJWsw9rFu9v80Pj1IpoHnwRcy6JwkoM1W1oRJTaOG9Vm8FpckQBmpVLscCJfNFrSPBz6kuscstygv_sGX5eVDoYD=w1805-h948-no?authuser=0)

## References:

- https://github.com/ivanovitchm/ppgeecmachinelearning

- https://github.com/ivanovitchm/colab2mlops

- https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents

- https://wandb.ai/site

- https://id.heroku.com/login

- https://www.anaconda.com/
