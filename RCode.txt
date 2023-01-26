#Since this is a segmentation task, we will use clustering to summarize customer segments

#First let us import and handle our data

getwd()
PATH ='marketing_campaign.csv'
df = read.csv(PATH,header=TRUE,sep='',stringsAsFactors = FALSE)

#Let us take a peak at our data

summary(df)
class(df)
View(df)

#Let us import needed libraries, BECAREFUL you have to install these packages first if not already installed, then you can import them

library(dplyr)
library(animation)
library(ggplot2)
library(stringr)
library(tidyr)
library(RColorBrewer)

#Let us check for noise

sum(is.na(df))

#Let us remove noise

df =na.omit(df)

#Data set up and trimming

df$Age = 2022 - df$Year_Birth

df$Education[df$Education == "2n Cycle"] = "UG"
df$Education[df$Education == "Basic"] = "UG"
df$Education[df$Education == "Graduation"] = "PG"
df$Education[df$Education == "Master"] = "PG"
df$Education[df$Education == "PhD"] = "PG"

df$Marital_Status[df$Marital_Status == "Divorced"] = "Single"
df$Marital_Status[df$Marital_Status == "Absurd"] = "Single"
df$Marital_Status[df$Marital_Status == "YOLO"] = "Single"
df$Marital_Status[df$Marital_Status == "Widow"] = "Single"
df$Marital_Status[df$Marital_Status == "Together"] = "Couple"
df$Marital_Status[df$Marital_Status == "Married"] = "Couple"
df$Marital_Status[df$Marital_Status == "Alone"] = "Single"

df$Customer_year= str_sub(df$Dt_Customer,-4)
df$Customer_year =as.numeric(df$Customer_year)
df$Customer_Seniority = 2022 - df$Customer_year

print(str(df))
#Becareful, here I had to convert df$Teenhome to integer from char to be able to use a binary operator and I knew its data type using the previous line
df$Child = df$Kidhome + as.integer(df$Teenhome)

df$Amt_Spent = df$MntWines + df$MntFishProducts + df$MntFruits + df$MntGoldProds + df$MntMeatProducts + df$MntSweetProducts
df$Num_Purchases_made= df$NumWebPurchases + df$NumCatalogPurchases + df$NumStorePurchases

#Thus we are left with 12 variables that can be used to create the distance matrix.

df =df[c(1,30,3,4,5,33,32,9,34,35,16,20)]

#Let us check our data after modification
View(df)

#Let us make all our columns integers or numerical to feed them to the k-model as it only needs integers to be able to calculate the Euclidean Distances
data2 = df
data2$Education = unclass(as.factor(data2$Education))
data2$Marital_Status = unclass(as.factor(data2$Marital_Status))
data2$Education = as.numeric(data2$Education)
data2$Income = as.numeric(data2$Income)
data2$Recency = as.numeric(data2$Recency)
data2$Marital_Status = as.numeric(data2$Marital_Status)

#Let us scale our data to same scale ;normalization

rescale_df =data2 %>%
  mutate(sID = scale(ID),
         sAge = scale(Age),
         sEducation = scale(Education),
         sMarital_Status = scale(Marital_Status),
         sIncome = scale(Income),
         sChild = scale(Child),
         sCustomer_Seniority=scale(Customer_Seniority),
         sRecency=scale(Recency),
         sAmt_Spent=scale(Amt_Spent),
         sNum_Purchases_made=scale(Num_Purchases_made),
         sNumDealsPurchases=scale(NumDealsPurchases),
         sNumWebVisitsMonth=scale(NumWebVisitsMonth)) %>%
select(-c(ID, Age, Education,Marital_Status, Income, Child, Customer_Seniority,Recency,Amt_Spent,Num_Purchases_made,NumDealsPurchases,NumWebVisitsMonth))

View(rescale_df)
#Let us check for noise

sum(is.na(rescale_df))

#Let us remove noise

df =na.omit(rescale_df)

#Final data ready for training
View(df)

#Let us start our model

#Training the model:
set.seed(2345)
kmeans.ani(df[2:3], 3)

#Evaluation Methods
#Contrary to supervised learning where we have the ground truth to evaluate the model's performance, clustering analysis doesn't have a solid 
#evaluation metric that we can use to evaluate the outcome of different clustering algorithms. Moreover, since kmeans requires k as an input and 
#doesn't learn it from data, there is no right answer in terms of the number of clusters that we should have in any problem. Sometimes domain 
#knowledge and intuition may help but usually that is not the case. In the cluster-predict methodology, we can evaluate how well the models are 
#performing based on different K clusters since clusters are used in the downstream modeling.

#To know our best k,use elbow method
kmean_withinss =function(k) {
  cluster = kmeans(df, k)
  return (cluster$tot.withinss)
}

## Try with 2 cluster
kmean_withinss(2)

# Set maximum cluster 
max_k =20 
# Run algorithm over a range of k 
wss = sapply(2:max_k, kmean_withinss)

# Create a data frame to plot the graph
elbow =data.frame(2:max_k, wss)

# Plot the graph with gglop
ggplot(elbow, aes(x = X2.max_k, y = wss)) +
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(1, 20, by = 1))

#Examining the cluster the best k is 7
model=kmeans(df, 7)

# Print the results

print(model)

#kmeans() function returns a list of components, including:
#cluster: A vector of integers (from 1:k) indicating the cluster to which each point is allocated
#centers: A matrix of cluster centers (cluster means)
#totss: The total sum of squares (TSS), i.e ???(xi???x¯)2. TSS measures the total variance in the data.
#withinss: Vector of within-cluster sum of squares, one component per cluster
#tot.withinss: Total within-cluster sum of squares, i.e. sum(withinss)
#betweenss: The between-cluster sum of squares, i.e. totss???tot.withinss
#size: The number of observations in each cluster
#iter: is the number of times the algorithm will repeat the cluster assignment and moving of centroids; The number of (outer) iterations.
#ifault: indicator of a possible algorithm problem -- for experts

#What are your clusters and their centres?

print(model$cluster)

print(model$centers)

print(model$totss)

print(model$withinss)

print(model$tot.withinss)

print(model$betweenss)

print(model$size)

print(model$iter)

print(model$ifault)

#If the manhattan distance metric is used in k-means clustering, 
#the algorithm still yields a centroid with the median value for each dimension, 
#rather than the mean value for each dimension as for Euclidean distance.
#Here if we check the model, we will find it has calculated the mean so

#What is your error function? Euclidian distance 

#Now let us plot our clusters

#Let's create the reshape dataset



# create dataset with the cluster number
center=model$centers
cluster = c(1: 7)
center_df = data.frame(cluster, center)

# Reshape the data

center_reshape = gather(center_df, features, values,sID:sNumWebVisitsMonth)
head(center_reshape)

#The code below create the palette of colors I will use to plot the heat map.

# Create the palette
hm.palette=colorRampPalette(rev(brewer.pal(10, 'RdYlGn')),space='Lab')

#You can plot the graph and see what the clusters look like.

# Plot the heat map
ggplot(data = center_reshape, aes(x = features, y = cluster, fill = values)) +
  scale_y_continuous(breaks = seq(1, 7, by = 1)) +
  geom_tile() +
  coord_equal() +
  scale_fill_gradientn(colours = hm.palette(90)) +
  theme_classic()
