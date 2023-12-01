# Problem Statement: Predicting Healthcare Costs based on Patient Attributes

from statistics import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

health = pd.read_csv('healthcare_dataset.csv')

#SUMMARY STATISTICS 
def get_summary_statistics(data):
    return data.describe()

# Usage
summary_stats = get_summary_statistics(health)
print(summary_stats)


#NUMBER OF ROWS AND COLUMNS
def get_shape(data):
    return data.shape


# Usage
rows, columns = get_shape(health)
print(f'Number of rows: {rows}, Number of columns: {columns}')


#COLUMN NAMES
def get_column_names(data):
    return data.columns.tolist()

# Usage
column_names = get_column_names(health)
print(f'Column names: {column_names}')




# Check for missing values in the entire dataset
missing_values = health.isnull().sum()
# Print the missing values (if any)
print("Missing Values:\n", missing_values)
# Check if there are any missing values in the dataset
if missing_values.sum() == 0:
    print("No missing values in the dataset.")
else:
    print("There are missing values in the dataset.")





# Display unique values for each column
for column in health.columns:
    unique_values = health[column].unique()
    print(f"Unique values in {column}:\n{unique_values}\n")




#Data Visualization 
#Graph showing the relationship between gender and age
# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Age', data=health, palette='pastel')

# Set plot labels and title
plt.xlabel('Gender')
plt.ylabel('Age')
plt.title('Distribution of Age by Gender')

# Show the plot
plt.show()


#Graph showing the relationship between gender and bloodtype 
# Create a grouped bar chart
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', hue='Blood Type', data=health, palette='pastel')

# Set plot labels and title
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of Blood Type by Gender')

# Show the plot
plt.legend(title='Blood Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#graph showing the relationship between gender and medical condition
# Create a grouped bar chart
plt.figure(figsize=(12, 6))
sns.countplot(x='Gender', hue='Medical Condition', data=health, palette='viridis', edgecolor='w')

# Set plot labels and title
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of Medical Conditions by Gender')

# Show the plot
plt.legend(title='Medical Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#Relationship between insurance and billing amount 
# Create a bar chart
plt.figure(figsize=(14, 8))
sns.barplot(x='Insurance Provider', y='Billing Amount', data=health, ci=None, palette='muted')

# Set plot labels and title
plt.xlabel('Insurance Provider')
plt.ylabel('Billing Amount')
plt.title('Relationship between Insurance Provider and Billing Amount')

# Show the plot
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()



#Relationship between medical condition and billing amount 

plt.figure(figsize=(14, 8))
sns.barplot(x='Medical Condition', y='Billing Amount', data=health, ci=None, palette='muted')

# Set plot labels and title
plt.xlabel('Insurance Provider')
plt.ylabel('Billing Amount')
plt.title('Relationship between Medical Condition and Billing Amount')

# Show the plot
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()


# #Creating the model 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd


# Specify the columns to be one-hot encoded
categorical_columns = ['Gender', 'Medical Condition', 'Insurance Provider', 'Blood Type']

# Feature selection (X) and target variable (y)
X = health[['Age'] + categorical_columns]
y = health['Billing Amount']

# Convert categorical variables to numerical representations (one-hot encoding)
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Additional Metrics and Confusion Matrix
# Convert predictions to binary (you might need to adjust the threshold)
threshold = 15000
binary_predictions = np.where(predictions > threshold, 1, 0)
binary_y_test = np.where(y_test > threshold, 1, 0)

# Calculate and print accuracy, precision, recall, and F1 score
accuracy = accuracy_score(binary_y_test, binary_predictions)
precision = precision_score(binary_y_test, binary_predictions)
recall = recall_score(binary_y_test, binary_predictions)
f1 = f1_score(binary_y_test, binary_predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Create confusion matrix
conf_matrix = confusion_matrix(binary_y_test, binary_predictions)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



