

import  numpy as np
import  matplotlib.pyplot as plt
import  re
import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from scipy.stats import pointbiserialr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import boxcox
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


np.random.seed(42)
#start region
def extract_date_summation(url):
    date_pattern = r'/(\d{4})/(\d{2})/(\d{2})/'
    match = re.search(date_pattern, url)
    if match:
        # Extract year, month, and day from the matched groups
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
    return (year*365+month*30+day)#date extraction function
def normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_minmax = (matrix - min_val) / (max_val - min_val)
    return normalized_minmax#matrix normalization#helper functions

def mean(vector):
    sum=0
    for i in range(len(vector)):
        sum+=vector[i]
    return sum/len(vector)
def variance(vector):
    meu=mean(vector)
    sum=0
    for i in range(len(vector)):
        sum+=(vector[i]-meu)**2
    return sum/len(vector)
def std(vector):
    return np.sqrt(variance(vector))
def polynomial_transformation(vector,degree):
    matrix=[]
    for i in range(degree):
        matrix.append(np.power(vector,i))
    return matrix
def mape(y_true,y_pred):
    avg_error=0
    for i in range(len(y_true)):
        error=np.abs(y_true[i]-y_pred[i])/np.abs(y_true[i])
        avg_error+=error
    avg_error/=len(y_true)
    return avg_error
#end region#TODO:HELPER FUNCTIONS
#start region
DATA=pd.read_csv("OnlineArticlesPopularity.csv")
DATA=DATA[DATA['channel type']!='[]']
PREPROCESSED_DATA={}

#end region#TODO:READING DATA

#start region
temp=[]
for url in DATA['url'].tolist():
    temp.append(extract_date_summation(url))

PREPROCESSED_DATA['normalized url date']=normalize(temp)
#end region#TODO:PREPROCESSING URL COLUMN
#start region

print("Total Elements Count in Title Column:"+np.size(DATA['title'].unique()).__str__())
print("Total Unique Elements Count in Title Column:"+DATA['title'].count().__str__()) #each elemnt in the column is unique...
#i will give each element a numerical value since it doesn't make any sense to make
# specified column for each unique element! also im dropping it later... ;)
temp=[]
for i in range(DATA['title'].count()):
    temp.append(i)
#end region#TODO:TITLE COLUMN PREPROCESSING
#start region

column_name="channel type"
null_symbol='[]'
total_count=np.sum(DATA[column_name].value_counts())
null_count=0
percentage=null_count/total_count*100
print("Null Percentage in channel type column:"+percentage.__int__().__str__()+"%")
#quite a lot 15% ... can't be removed must be replaced...
#what about replacing according the probability distribution
non_null_count=total_count-null_count
count_distribution={}
choice_dictionary={}
choice_number=0
for string in DATA[column_name].value_counts().keys():
    if string!=null_symbol:
        element_count=DATA[column_name].value_counts()[string]
        current_probability=(element_count/non_null_count)
        count_distribution[string]=(current_probability*null_count).__int__()
        if string==' data_channel_is_world':
            count_distribution[string]+=3#account for missing entries due to integer casting...
        choice_dictionary[choice_number]=string
        choice_number+=1
#time to make the encoding

channel_type_encoding_matrix={}
for string in DATA[column_name].value_counts().keys():
    if string!=null_symbol:
        channel_type_encoding_matrix[string]=[]

for string in DATA[column_name]:
    if string!=null_symbol:
        for str in channel_type_encoding_matrix.keys():
            if str!=string:
                channel_type_encoding_matrix[str].append(0)
        channel_type_encoding_matrix[string].append(1)

    elif string==null_symbol:
        #make a random choice
        vector_of_choices=[key for key in choice_dictionary.keys()]
        choice=np.random.choice(vector_of_choices)
        string_choice=choice_dictionary[choice]
        channel_type_encoding_matrix[string_choice].append(1)
        count_distribution[string_choice]-=1
        if(count_distribution[string_choice]==0):
            del  choice_dictionary[choice]
        for str in channel_type_encoding_matrix.keys():
            if str!=string_choice:
                channel_type_encoding_matrix[str].append(0)


#end region #TODO: CHANNEL TYPE COLUMN PREPROCESSING

#start region

column_name="weekday"
weekday_encoding_matrix={}
for string in DATA[column_name].unique():
    weekday_encoding_matrix[string]=[]
for string in DATA[column_name]:
    weekday_encoding_matrix[string].append(1)
    for str in weekday_encoding_matrix.keys():
        if str != string :
            weekday_encoding_matrix[str].append(0)


#end region#TODO:WEEKDAY COLUMN PREPROCESSING

#start region
column_name="isWeekEnd"
isWeekEnd_encoding_matrix={"isWeekEnd":[]}
for string in DATA[column_name]:
    if string == "Yes":
        isWeekEnd_encoding_matrix["isWeekEnd"].append(1)
    else :
        isWeekEnd_encoding_matrix["isWeekEnd"].append(0)

#end region #TODO:ISWEEKEND COLUMN PREPROCESSING

BINARY_COLUMNS={}
for key in channel_type_encoding_matrix.keys():
    BINARY_COLUMNS[key]=channel_type_encoding_matrix[key]
for key in weekday_encoding_matrix.keys():
    BINARY_COLUMNS[key]=weekday_encoding_matrix[key]





#TODO:------------------------------------STRING COLUMNS PREPROCESSING DONE-----------------------------------


NUMERICAL_COLUMNS={}
def is_numerical_column(string_column):
    string_vector=['channel type','url','title','weekday','isWeekEnd']
    for string in string_vector:
        if string==string_column:
            return False
    return True


for string_column in DATA.columns:

    if(is_numerical_column(string_column)):
        #if string_column == ' shares':
         #   continue
        NUMERICAL_COLUMNS[string_column]=DATA[string_column].tolist()
#TODO:OUTLIER HANDLING
threshold=3
for string_key in NUMERICAL_COLUMNS.keys():

    meu=mean(NUMERICAL_COLUMNS[string_key])
    deviation=std(NUMERICAL_COLUMNS[string_key])
    #upper=np.percentile(NUMERICAL_COLUMNS[string_key],75)
    #lower=np.percentile(NUMERICAL_COLUMNS[string_key],25)
    upper=meu+threshold*deviation
    lower=meu-threshold*deviation
    for i in range(len(NUMERICAL_COLUMNS[string_key])):
        if NUMERICAL_COLUMNS[string_key][i]>upper:
            NUMERICAL_COLUMNS[string_key][i]= upper
        if NUMERICAL_COLUMNS[string_key][i]<lower:
            NUMERICAL_COLUMNS[string_key][i]= lower

best_numerical_columns=30
best_binary_columns=13
binary_correlation_vector=[]
binary_correlation_dict={}
correlation_vector=[]
correlation_dict={}
for key in BINARY_COLUMNS.keys():
    binary_correlation=pointbiserialr(BINARY_COLUMNS[key],DATA[' shares'])[0]
    binary_correlation_dict[binary_correlation]=key
    binary_correlation_vector.append(binary_correlation)
binary_correlation_vector=sorted(binary_correlation_vector,key=abs,reverse=True)

for key in NUMERICAL_COLUMNS.keys():
    if key==' shares':
        continue
    current_correlation=DATA[key].corr(DATA[' shares'])
    correlation_dict[current_correlation]=key
    correlation_vector.append(current_correlation)
correlation_vector=sorted(correlation_vector,key=abs,reverse=True)
for i in range(best_numerical_columns):
    key=correlation_dict[correlation_vector[i]]
    PREPROCESSED_DATA[key]=NUMERICAL_COLUMNS[key]
for i in range(best_binary_columns):
    key=binary_correlation_dict[binary_correlation_vector[i]]
    PREPROCESSED_DATA[key]=BINARY_COLUMNS[key]
PREPROCESSED_DATA=pd.DataFrame(PREPROCESSED_DATA)
print(np.shape(PREPROCESSED_DATA))
Y=DATA[' shares'].to_numpy()
Y=boxcox(Y)[0]

x_train,x_test,y_train,y_test=train_test_split(PREPROCESSED_DATA.to_numpy(),Y,test_size=0.15,random_state=1)
#TODO:-------------------------------LINEAR REGRESSION---------------------------------------------
linear_model=LinearRegression()
linear_model.fit(x_train, y_train)
linear_predictions=linear_model.predict(x_test)
linear_accuarcy=100-mape(y_true=y_test,y_pred=linear_predictions)
linear_R_score=r2_score(y_true=y_test,y_pred=linear_predictions)
print("Linear Accuracy:"+linear_accuarcy.__str__()+"%")
print("Linear R Squared Score:"+linear_R_score.__str__())


#TODO:-------------------------------LINEAR REGRESSION---------------------------------------------

#TODO:-------------------------------RIDGE---------------------------------------------
alpha = 1 # Regularization strength
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(x_train, y_train)
ridge_predictions = ridge_model.predict(x_test)
ridge_accuarcy=100-mape(y_true=y_test,y_pred=ridge_predictions)
ridge_R_score=r2_score(y_true=y_test,y_pred=ridge_predictions)
print("Ridge Accuracy:"+ridge_accuarcy.__str__()+"%")
print("Ridge R Squared Score:"+ridge_R_score.__str__())

#TODO:-------------------------------RIDGE---------------------------------------------


#TODO:----------------------------------GRADIENT BOOSTING MODEL---------------------------------------------------
gradient_boosting_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=1)
gradient_boosting_model.fit(x_train, y_train)
gradient_predictions = gradient_boosting_model.predict(x_test)
gradient_accuarcy=100-mape(y_true=y_test,y_pred=gradient_predictions)
gradient_R_score=r2_score(y_true=y_test,y_pred=gradient_predictions)
print("Gradient Boosting Accuracy:"+gradient_accuarcy.__str__()+"%")
print("Gradient Boosting R Squared Score:"+gradient_R_score.__str__())

#TODO:----------------------------------GRADIENT BOOSTING MODEL---------------------------------------------------
#rf_regressor = RandomForestRegressor(n_estimators=50, random_state=1)

# # Train the model
# rf_regressor.fit(x_train, y_train)
#
# # Make predictions on the testing set
# rf_regressor_predictions = rf_regressor.predict(x_test)
# rf_accuarcy=100-mape(y_true=y_test,y_pred=rf_regressor_predictions)
# rf_R_score=r2_score(y_true=y_test,y_pred=rf_regressor_predictions)
# print("Random Forest Accuracy:"+gradient_accuarcy.__str__()+"%")
# print("Random Forest R Squared Score:"+gradient_R_score.__str__())
# for i in range(len(x_test)):
#     plt.scatter(np.transpose(x_test)[i],y_test,s=1.5,label="TEST DATA")
#     plt.scatter(np.transpose(x_test)[i],rf_regressor_predictions,color='red',s=1.5,label="MODEL OUTPUT")
#     plt.legend()
#     plt.show()