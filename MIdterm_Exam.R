## BANL6900 Business Analytics Capstone Midterm Exam

# Name: Anish Gangur Kaushik

### 1. *Load the dataset*:

rental_data = read.csv("Downloads/RentalProperties.csv")


### 2. *Question 1: Exploratory Data Analysis (EDA):

# Display the structure of the dataset
str(rental_data)

# Display the first few rows of the dataset
head(rental_data)

# Check for missing values
sum(is.na(rental_data))

# Display summary statistics for each variable
summary(rental_data)


#*Explanation*: 
# str() gives an overview of the data structure, variable types, and sample values.
# head() shows the first few rows of the data to visually inspect how it looks.
# summary() provides summary statistics for numerical columns.

#*List any issues*: Mention missing values, strange data entries (like negative values where they shouldn’t be), or format issues in your notes.

# Missing or Non-Numeric Data in Numeric Columns:
# The floor, animal, furniture, monthlyFee, rentAmount, propertyTax, fireInsurance columns have $ signs present.
# These columns need to be cleaned and converted to numeric format to avoid issues in analysis.

# Outliers in the Area Column:
# There are some outliers in the area column that need to be handled properly.
# The area column seems heavily skewed due to the large values, which could affect statistical analyses.

# Investigation of Rooms and Bathroom Columns:
# The rooms and bathroom columns have maximum values of 10, which although possible, might need further investigation
# to ensure there are no input errors.


### 3. *Question 2: Business Questions*:
#- *Normal Question: A typical business question might be, "What factors predict the rent price of a property?" We could choose *target* variable: rent price and *predictors* as: area, rooms, bathrooms, parking.
#- *Proxy Unusual Question: A more unusual question could be, "Does having furniture affect the number of rooms in a property?" *target variable: number of rooms, **predictors*: furniture (whether the property is furnished).

### 4. *Question 3: Methods*:
#- We need to suggest methods based on the questions proposed in Question 2.

#- For *Normal Question: We could use **linear regression* since the target variable is continuous.
#- For *Proxy Question: We could use **logistic regression* if the target variable is categorical or *classification models* like decision trees.

### 5. *Question 4: Visualizations and Correlations*:

# Histogram for each variable
par(mfrow=c(1,1))
hist(rental_data$area, main = "Histogram of Area", xlab = "Area")
dev.off()
#Comment: The histogram of the area variable is right-skewed, indicating that most properties have smaller areas, 
# with a few properties having much larger areas, as shown by the long tail to the right.

hist(rental_data$rooms, main = "Histogram of Rooms", xlab = "Rooms")
# Comment: The distribution of rooms is right-skewed, showing that most properties have between 1 to 3 rooms, 
# with fewer properties having more than 4 rooms.

hist(rental_data$bathroom, main = "Histogram of Bathrooms", xlab = "Bathrooms")
# Comment: The distribution of bathrooms is right-skewed, indicating that most properties have 1 to 3 bathrooms, 
# with fewer properties having more than 3 bathrooms.

hist(rental_data$parking, main = "Histogram of Parking", xlab = "Parking")
# Comment: The distribution of parking is right-skewed, showing that most properties have between 0 to 2 parking spots,
# with a few properties having more than 2 spots.

# Boxplot for area vs furniture
boxplot(rental_data$area ~ rental_data$furniture, main = "Area vs Furniture", ylab = "Area")
#There were many outliers when the above code was run, hence proceeding to next step to take median

# Replace two highest outliers in area with the median
rental_data$area = ifelse(rental_data$area == max(rental_data$area), median(rental_data$area), rental_data$area)

# Boxplot for area vs furniture after adjusting outliers
boxplot(rental_data$area ~ rental_data$furniture, main = "Area vs Furniture", ylab = "Area")

# Correlation matrix for area, rooms, bathroom, and parking
cor_matrix = cor(rental_data[, c("area", "rooms", "bathroom", "parking")])
print(cor_matrix)

# Scatter plot matrix
pairs(rental_data[, c("area", "rooms", "bathroom", "parking")])


# Question 4:

#e)	If you were going to predict number of parking spots, which two predictors would you suggest? Why?

# 1. Area: It has a moderately strong correlation with parking (correlation = 0.770).
#    As the property area increases, the number of parking spots tends to increase.
#    Larger properties often have more space to accommodate additional parking.

# 2. Bathroom: It has a strong correlation with parking (correlation = 0.686).
#    Properties with more bathrooms tend to be larger and more luxurious, which could explain the increase in parking spaces.

### 6. *Question 5: Regression Model*:

# Linear regression model
model = lm(rooms ~ area + bathroom + parking, data = rental_data)

# Check statistical significance and R-squared value
options(scipen = 999)
summary(model)


# Intercept (1.1602):
# The intercept is statistically significant with a p-value of < 2e-16.
# This means that when area, bathroom, and parking are all zero (which may not be realistic in a practical context), 
# the expected baseline value for rooms is approximately 1.16.

# Area (0.0022):
# The coefficient for area is 0.0022, meaning for each unit increase in the area (keeping other variables constant), 
# the number of rooms is expected to increase by 0.0022.

# Bathroom (0.3649):
# The coefficient for bathroom is 0.365, meaning for each additional bathroom, the number of rooms is expected to increase by about 0.365,
# assuming other variables are held constant.

# Parking (0.0882):
# The coefficient for parking is 0.088, meaning for each additional parking spot, the number of rooms is expected to increase by about 0.088,
# assuming other variables are held constant.

# Multiple R-squared: 0.6039:
# This means that 60.39% of the variability in the number of rooms is explained by the model (i.e., by area, bathroom, and parking).
# This is a fairly good R-squared value, indicating that the model explains a significant portion of the variance in the number of rooms.

# Adjusted R-squared: 0.6038:
# The adjusted R-squared is very close to the multiple R-squared, indicating that the model is a good fit.



# Predict rooms for a given property
new_property = data.frame(area = 100, bathroom = 3, parking = 4)
predict(model, new_property)

