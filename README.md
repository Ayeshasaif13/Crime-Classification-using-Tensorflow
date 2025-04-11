# Crime-Classification_Using_PySpark

## Summary of the Project
This project focuses on **San Francisco Crime Classification** by leveraging the power of PySpark to process and analyze crime data. In modern law enforcement, data plays a crucial role in informing strategies and responses. This repository demonstrates how to utilize PySpark to handle data efficiently and train machine learning models for identifying cybercrimes in the San Francisco region.

## Motivation
The main objective is to develop an automated approach capable of categorizing different cyber-related offenses based on provided descriptions. Quick and accurate classification of criminal activities enables law enforcement agencies to act faster and implement appropriate countermeasures.

## Core Elements

1. **Data Acquisition**  
   - Network packet captures are gathered from various sources around San Francisco. The raw data includes essential network details such as IP addresses, ports, protocols, and packet contents.

2. **Data Cleansing & Preparation**  
   - We use PyShark to filter, convert, and sanitize the raw packet capture data. This step involves cleaning inconsistencies, ensuring data quality, and extracting only the features needed for subsequent analysis.

3. **Feature Extraction**  
   - Relevant attributes from the captured packets (e.g., protocol specifics, payload characteristics) are selected or transformed to form a reliable feature set. These engineered features serve as the input for the classification model.

4. **Machine Learning Model Setup**  
   - Various algorithms (e.g., neural networks, random forests) may be used for training on the preprocessed dataset. The primary goal is to categorize network traffic into different cyber-incident types (e.g., DDoS, malware, phishing).

5. **Performance Analysis**  
   - Several evaluation metrics (accuracy, precision, recall, F1-score) are calculated to measure the effectiveness of the model. Techniques such as cross-validation help verify robustness and guard against overfitting.

## Where to Find the Data
A sample dataset can be found on Kaggle at the link below:  
[San Francisco Crime Classification Dataset](https://www.kaggle.com/competitions/sf-crime/data?select=train.csv.zip)

