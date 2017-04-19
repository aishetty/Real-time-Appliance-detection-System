# Real-time-Appliance-detection-System

The project is aimed at identifying the active devices at any given point in time using a Non-Intrusive approach of detecting appliances.
The implementation is done on an Intel Edison development platform using a Clip on Current Transoformer and a Voltage tranformer. 

Step 1: To run the code 'appliance_detect.py' you will need the latest packages of 'sklearn', 'scipy', 'numpy', 'subprocess','socket' installed on the Intel Edison. 

Step 2: Once the 'Real-time-appliance-identification' folder is downloaded onto the Intel Edison , change the 'Data_path' variable to the corresponding path of the 'Real-time-appliance-identification' folder.

Step 3: Please change the frequency of sampling 'fs' according to your sampling rate.

Step 4: As the training data needs to be stored in the 'CSV' folder, as the csv files are stored, change the meta.json file correspondingly.

Step 5: Once, the arduino code runs on Intel Edison, the test data will be changed and the python code sends the active devices to the Android smartphone through socket programming when both Intel edison and Smartphone are in the same WLAN. 
