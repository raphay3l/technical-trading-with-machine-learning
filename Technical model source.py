"""
Designed and implemented by Rafal Jankowski, Apr 2016. 

Starter code to explore technical trading methods with machine learning.
Price Data.xlsx is loaded into DataFrame in which contains technical indicators calculated on a flat price of WTI futures contract.
We are experimenting with two types of predictions:
- Classification - i.e. if a futures price some N periods from now is higher/lower from the current price (implemented as N = 1 below)
- Regression - calculate a simple future return r = P(t + N) / P(t) of buying today and holding for N periods (currently implemented
as simply regression on a futures price at next period P(t + 1))

Technical indicators are correlated with one-another hence the simple regression would fail due to multi-correlation problem.

PCA analysis informs us that just a few technical indicators explain most of the variation in the parameters. Learning on the 
whole set of P(t - 1), P(t - 2)... etc. would be impractical. 

Application of this method would be to train a 'fair' price based on fundamental indicators and not just technical ones.

SVR, Bagger and Gradient Boosting Regressors are analyzed and compared.
"""

from sknn.mlp import Regressor, Classifier, Layer
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

import xlwt
from tempfile import TemporaryFile


# load data
xl = pd.ExcelFile('Price Data.xlsx')
data  = xl.parse('Data')
data = data[70:]
del data['Date / Time']

# list of headers (technical indicators)
list_heads = [column for column in data]
for i in range(0, len(list_heads) - 1):
    data[list_heads[i]] = pd.to_numeric(data[list_heads[i]], errors = 'coerce' )

# select specific columns for gasoline crack
data_temp = pd.DataFrame()
Y = pd.DataFrame()
Y2 = pd.DataFrame()

# list of all indicators (to match in the supporting excel file)
indicators = ["Trade Close 2", "ATR_20", "Bollinger_up", "Bollinger_down", "Chandelier_long", "Chandelier_short", "Ichimoku_Cloud", \
            "SMA_5", "SMA_10", "SMA_15", "SMA_20","SMA_30","SMA_45","SMA_60","Aroon_up_21","Aroon_down_21", "EMA_10", "EMA_20", \
            "EMA_45", "ROC_diff", "Coppock_Curve"]

#or select sub-group
#indicators = ["Trade Close 2", "SMA_5", "SMA_10", "SMA_15",  "EMA_10", "EMA_20"]
            
# Two columns - one for classification another for regression
# classificiation is simply question whether next period's price is higher/lower than current's
# but can be used (and ought to be) on return forecast r = P(t + n) / P(t) for some n periods in the future
y_column = 'Y1' 
y_column2 = 'Y2'

for i in range(0, len(indicators)):
    #data[list_heads[i]] = pd.to_numeric(data[list_heads[i]], errors = 'coerce' )
    data_temp[indicators[i]] = pd.to_numeric(data[indicators[i]], errors = 'coerce' )
    data_temp[indicators[i]] = pd.to_numeric(data[indicators[i]], errors = 'coerce' )
X = data_temp

Y[y_column] = data[y_column]
Y2[y_column2] = data[y_column2]

#remove NANs
for i in range(0, len(indicators)):
    X = X[np.isfinite(X[indicators[i]])]    

y2= np.array(Y2[y_column2]).astype("int")
y= np.array(Y[y_column])#.astype("int"))
x= np.array(X)

# -------------------- train best fit ----------------------------------

train_size = round(len(x) * 0.8,0)
test_size = len(x) - train_size
train_sizes=np.linspace(1, test_size, test_size)

# PCA analysis -----------------------------
pca = decomposition.PCA(n_components = 5)
pca.fit(x)
X_reduced = pca.fit_transform(scale(X))
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

# select reduced X vector
#x= X_reduced
# --------------------------------------------------
estimator = SVR(kernel="linear")
selector = RFE(estimator, 3, step=1)
selector = selector.fit(x[:5000], y[:5000])
selector.ranking_


# -------------------------------------------------- REGRESSORS -------------------------------------------------------

# ---------------- analyze fits ------------------------
# Set up SVR, Bagger and Gradient Boosting Regressors
svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
bagger = BaggingRegressor()
GBR = GradientBoostingRegressor()
#kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1) # Kernel ridge does not work well

# Train on different lots sizes
train_sizes, train_scores_svr, test_scores_svr = \
    learning_curve(svr, x[:train_size], y[:train_size], train_sizes=np.linspace(0.1, 1, 5),
                   scoring="mean_squared_error", cv=10)
train_sizes_abs, train_scores_GBR, test_scores_kr = \
    learning_curve(GBR, x[:train_size], y[:train_size], train_sizes=np.linspace(0.1, 1, 5),
                   scoring="mean_squared_error", cv=10)
train_sizes_abs, train_scores_bagger, test_scores_bagger = \
    learning_curve(bagger, x[:train_size], y[:train_size], train_sizes=np.linspace(0.1, 1, 5),
                   scoring="mean_squared_error", cv=10)

# plot results
plt.plot(train_sizes, test_scores_svr.mean(1), 'o-', color="r",
         label="SVR")
plt.plot(train_sizes, train_scores_GBR.mean(1), 'o-', color="g",
         label="GBR")
plt.plot(train_sizes, test_scores_bagger.mean(1), 'o-', color="b",
         label="bagger")
         
plt.xlabel("Train size")
plt.ylabel("Mean Squared Error")
plt.title('Learning curves')
plt.legend(loc="best")
# ---------------------------------------

# Two best regressors are Bagger and SVR - set these up for Excel run on the full testing lot
bagger = BaggingRegressor()
bagger.fit(x[:train_size], y[:train_size]) 
y_bag = bagger.predict(x[train_size:])
y_bag_full = bagger.predict(x)

svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
svr.fit(x[:train_size], y[:train_size]) 
y_svr = svr.predict(x[train_size:])
y_svr_full = svr.predict(x)


# -------------------------------------------------- Classifiers -------------------------------------------------------
# Two best classifiers are Bagger and Gaussian - set these up for Excel run on the full testing lot
bagger_class = BaggingClassifier()
bagger_class.fit(x[:train_size], y2[:train_size])
y_bag_class = bagger_class.predict(x)

gnb_class = GaussianNB()
gnb_class.fit(x[:train_size], y2[:train_size]) 
y_gnb_class = gnb_class.predict(x)


# ---------------------------- PRINT TO EXCEL ----------------------------
book = xlwt.Workbook()

sheet1 = book.add_sheet('regressors')
sheet2 = book.add_sheet('classifiers')

export_data = [y, y_bag_full, y_svr_full]
export_data2 = [y, y_bag_class, y_bag_class]

for row, array in enumerate(export_data):
    for col, value in enumerate(array):
        sheet1.write(col, row, value)

for row, array in enumerate(export_data2):
    for col, value in enumerate(array):
        sheet2.write(col, row, value)
        
name = "output.xls"
book.save(name)
book.save(TemporaryFile())


#
#plt.scatter(train_sizes, y[train_size:], c='k', label='data')
#plt.hold('on')
##plt.plot(train_sizes, y_svr, c='r')
#plt.hold('on')
#plt.plot(train_sizes, y_bag, c='b')
#plt.hold('on')
##plt.plot(train_sizes, y_kr, c='g')
#plt.xlabel('data')
#plt.ylabel('target')
#plt.figure()


#set up neural net

