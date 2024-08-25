#####################################################
# Comparison of the Transformation of Bidding Methods with the help of the AB Test
#####################################################

#####################################################
# Explanation of Our problem
#####################################################

# Facebook recently introduced a new type of bidding, "average bidding", as an alternative to the existing type of bidding called "maximumbidding".
# One of our clients, bombabomba.com, decided to test this new feature and would like to run an A/B test to see if averagebidding brings more conversions than maximumbidding.
# The A/B test has been running for 1 month and bombabomba.com is now waiting for you to analyze the results of this A/B test.
# The ultimate measure of success for Bombabomba.com is Purchase. Therefore, for statistical testing, the focus should be on the Purchase metric.


#####################################################
# Dataset Story
#####################################################

# This dataset, which contains a company's website information, includes information such as the number of advertisements seen and clicked by users,
# as well as information about the earnings from these advertisements.
# There are two separate datasets, the Control and the Test group.
# These data sets are located on separate sheets of the excelab_testing.xlsx.
# Maximum Bidding was applied to the control group and Average Bidding was applied to the test group.

# impression: Number of ad views
# Click: Number of clicks on the displayed ad
# Purchase: Number of products purchased after clicked ads
# Earning: Earnings from purchased products

#####################################################
# Project Tasks
#####################################################

#####################################################
# DUTY 1:  Preparing and Analyzing Data
#####################################################

# Step 1:  Read the data set named ab_testing_data.xlsx which consists of control and test group data.
# Assign control and test group data to separate variables.


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import math
from  sklearn.preprocessing import MinMaxScaler
import scipy.stats as st
import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_control = pd.read_excel("/Users/macbook/Desktop/miul/Miuul/3. hafta meausemremnt problems/ödevler /ABTesti/ab_testing.xlsx", sheet_name="Control Group")
df_test =  pd.read_excel("/Users/macbook/Desktop/miul/Miuul/3. hafta meausemremnt problems/ödevler /ABTesti/ab_testing.xlsx", sheet_name="Test Group")
df_c = df_control.copy()
df_t = df_test.copy()
df_c.head()
df.shape
df_c.describe().T




# Step 2: Analyze the control and test group data.
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df_c)
check_df(df_t)


# Step 3: After the analysis, combine the control and test group data using the concat method.

df_t["group"] = "test"
df_c["group"] = "control"
df = pd.concat([df_c, df_t], axis = 0, ignore_index= False )
df.head()



#####################################################
# DUTY 2:  Defining the Hypothesis of the A/B Test
#####################################################

# Step 1: Define the hypothesis.

# H0 : M1 = M2 (There is no difference between control group and test group purchase averages.)
# H1 : M1!= M2 (There is a difference between control group and test group purchase averages.)


# Step 2: Analyze the purchase (gain) averages for the control and test group
df.groupby("group").agg({"Purchase": "mean"})



#####################################################
# DUTY 3: Performing Hypothesis Testing
#####################################################

# Step 1: Before conducting hypothesis testing, make assumption checks. These are the Assumption of Normality and Homogeneity of Variance.

# Test whether the control and test groups comply with the normality assumption separately on the Purchase variable
# Assumption of Normality :
# H0: The assumption of normal distribution is met.
# H1: Normal distribution assumption is not met
# p < 0.05 H0 rejection
# p > 0.05 H0 NOT REJECTABLE
# According to the test result, is the normality assumption met for the control and test groups?
# Interpret the p-value values obtained.


test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value=0.5891 > 0.05
# HO cannot be rejected. The values of the Control group satisfy the assumption of normal distribution.
test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value=0.1541 > 0.05
# HO cannot be rejected. The values of the test group satisfy the assumption of normal distribution.

# Variance Homogeneity
# H0: Variances are homogeneous.
# H1: Variances are not homogeneous.
# p < 0.05 H0 rejection
# p > 0.05 H0 not rejectable
# Test whether homogeneity of variance is achieved for the control and test groups on the Purchase variable.
# According to the test result, is the homogeneity assumption met? Interpret the p-values obtained.

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                           df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value=0.1083 > 0.05
# HO cannot be rejected. The values of Control and Test group satisfy the assumption of variance homeogeneity.
# Variances are Homogeneous





# Step 2: Select the appropriate test based on the Assumption of Normality and Homogeneity of Variance results

# Since the assumptions are met, an independent two sample t-test (parametric test) is performed.



# H0: M1 = M2 (There is no statistically significant difference between the control group and test group purchase averages.)
# H1: M1 != M2 (There is a statistically significant difference between the control group and test group purchase averages.)
# p<0.05 HO Rejection , p>0.05 HO Not Rejectable

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                              df.loc[df["group"] == "test", "Purchase"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value = 0.3493 > 0.05
# HO cannot be rejected

# Step 3: Considering the p_value obtained as a result of the test,
# interpret whether there is a statistically significant difference between the control and test group purchase averages.

# p-value=0.3493
# HO cannot be rejected. There is no statistically significant difference between control and test group purchase averages.





































