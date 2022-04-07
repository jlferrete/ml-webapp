from sklearn import linear_model
import pandas as pd
import pickle

df = pd.read_csv('prices.csv')

value = df['Value']  # dependent variable
rd = df[['Rooms', 'Distance']]  # independent variable

lm = linear_model.LinearRegression()
lm.fit(rd, value)  # fitting the model
pickle.dump(lm, open('model.pkl', 'wb'))  # save the model

print(lm.predict([[15, 61]]))  # format of input
print(f'score: {lm.score(rd, value)}')
