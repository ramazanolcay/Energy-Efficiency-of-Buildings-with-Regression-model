# Energy-Efficiency-of-Buildings-with-Regression-model
Benchmark of a lot of regression models

I have taken dataset from here: https://www.kaggle.com/elikplim/eergy-efficiency-dataset

Step 1- there were no column name in data and I added the names ("Relative Compactness","Surface Area","Wall Area",
                                                                        "Roof Area", "Overall Height","Orientation","Glazing Area",
                                                                        "Glazing Area Distribution", "Heating Load", "Cooling Load")
                                                                        
Step 2- I have check NaN values and result is in image

![image](https://user-images.githubusercontent.com/87130915/178059146-146b9c71-7ebf-4f75-9efb-6f74ae0e93ba.png)


Step 3- Splitin dataframe with test_size=0.33

Step 4- Trying linear regression, polynomial regression, Desicion Tree Regressor, Random Forest Regressor, Gaussian Process Regressor, PLS Regression and XGboost.

Step 5-Comparing metrics which are R2 and mse (Mean Squared Error)

Result are in image


![image](https://user-images.githubusercontent.com/87130915/178060189-81a0bb25-2167-4aaa-a65e-60794212e825.png)


XGboost giving the best result for data and Gaussian result was unexpected for me 
