# Affluency-of-localities-using-Mahalanobis-distance
Affluence of different localities of New Delhi (Rajendar Nagar, R.K. Puram, Karol Bagh) using three main features i.e. coffee shops,salon rates in locality and the rate of property in locality.

# Creating dataset
First we got the electoral roll of 3 mentioned constituencies and using OCR we converted it into excel dataset.From that we found out out different part numbers in each constituency.

Based on these part numbers we proceeded as follows :-
For finding out the affluence of a society we decided to choose rate of coffe shops and salon which have decent rates of minimum of Rs 150 for coffee and Rs 100 for a haircut.
Now to define the area we took a radius of 1 km for coffee shops and 400m for salon.
Third feature which we added was the rate of property.

We use GIS and other online platforms to locate coffee shops and salons part wise in our dataset. 

# Mahalnobis Distance
Using Mahanlobis distance we tried to find out the centroid point of different localities and assigned different weights to them on the basis of their property rates. Hence we found out the affluency for different parts in Rajendar nagar locality.

# Further Analysis
Now we added our Electoral roll dataset in above dataset and calculated different figures such as
1) Male-Female Count/Ratio
2) Married Female 
3) Young population based on EC of india
4) Young Married females

Other points such as old age people in locality,Young Male etc can also be found out using the dataset.

We saw different trends between affluency calculated using Mahanolobis distance and above pointswhich can be used to make effective policies based on the segment of population.

Acknowledgement:
Mr Akhilesh Ajith Kumar 
Mr Suyash Gulsti 
