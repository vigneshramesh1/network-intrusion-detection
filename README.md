# Detecting Intrusion in Softwarized 5G Networks Using Machine Learning

## Abstract
The emergence of 5G technology has transformed the wireless communication landscape, offering unprecedented data speeds and connectivity for numerous devices and applications. However, this technological advancement also introduces new cybersecurity vulnerabilities, making 5G networks an attractive target for sophisticated cyber-attacks. This paper addresses the critical need for robust intrusion detection systems (IDS) in softwarized 5G networks by leveraging advanced machine learning (ML) techniques. Utilizing the comprehensive [5G Network Intrusion Detection Dataset (5G-NIDD)](https://ieee-dataport.org/open-access/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless), which contains concatenated data encompassing all attack scenarios, we propose and evaluate several complex ML models aimed at accurately classifying network activities as either malicious or benign. Our research contributes to the ongoing efforts to secure 5G networks by providing insights into the effectiveness of ML-based IDS in detecting a wide range of cyber threats, thereby ensuring the integrity and reliability of 5G services.

## Overview
This repository contains the implementation and evaluation of machine learning models for intrusion detection in softwarized 5G networks. The code and datasets used in this research are provided to facilitate reproducibility and further exploration of the proposed methods.

## Dataset
For the purpose of our research, we utilized the comma-separated value (CSV) file from the 5G Network Intrusion Detection Dataset (5G-NIDD), which contains concatenated data encompassing all attack scenarios. This consolidated file provided a comprehensive representation of the network traffic data, including both malicious and benign activities. The dataset consists of 1,215,890 rows, with 477,737 representing benign activities and 738,153 representing various types of malicious attacks. Figure 1 shows the distribution of these benign and malicious activities present in the dataset. (Reference: Sehan Samarakoon, Yushan Siriwardhana, Pawani Porambage, Madhusanka Liyanage, Sang-Yoon Chang, Jinoh Kim, Jonghyun Kim, and Mika Ylianttila. 5g-nidd: A comprehensive network intrusion detection dataset generated over 5g wireless network, 2022.)

## Models Implemented
We implemented the following four models:
- Dense Neural Network with Autoencoder
- K-Nearest Neighbors (KNN)
- Random Forest (RF)
- Support Vector Machine (SVM)

## Getting Started
To get started with replicating our research findings, follow these steps:

1. Clone this repository to your local machine.
2. Download the 5G Network Intrusion Detection Dataset (5G-NIDD) from the provided source and place it in the `Dataset` directory.
3. Navigate to the `Models` directory and follow the instructions in the README file to run the machine learning models.
4. Refer to the `Results` directory for the outcomes of our experiments and performance evaluation.

## Contributing
Contributions to this research project are welcome. If you have suggestions for improvements or would like to collaborate, please feel free to open an issue or submit a pull request.

## Contact
For any inquiries or questions regarding this research, please contact [Abdallah Moubayed(abdallah.moubayed@asu.edu) or Vignesh Ramesh(vrames25@asu.edu)].
