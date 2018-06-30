"""
Author: Koushik Thai
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
from pathlib import Path
from fancyimpute import KNN, MICE as m
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score

"""
def latestFiveMoviesForStudio():    
    studioSet = set()
    for studio in rt['Studio']:
        if studio is not np.nan:
            studioSet.add(studio)
    for studio in studioSet:
        df2 = rt.loc[rt['Studio'] == studio]
        df1 = df2.sort_values('year', ascending=False).head(5)
        file_exists = os.path.isfile('predata12.csv')
        with open('predata12.csv','a', encoding='latin-1') as csvfile:
            if not file_exists:
                df1.to_csv(csvfile, header=True)
            else:
                df1.to_csv(csvfile, header=False)
"""
       
def scaleTheData(rt):
    normalizedDF = rt
    normalized = (normalizedDF['length']- min(normalizedDF['length']))/(max(normalizedDF['length']) - min(normalizedDF['length']))
    rt.update(normalized)
    #normalized1 = (normalizedDF['Fresh']- min(normalizedDF['Fresh']))/(max(normalizedDF['Fresh']) - min(normalizedDF['Fresh']))
    #rt.update(normalized1)
    #normalized1 = (normalizedDF['audience - User Ratings']- min(normalizedDF['audience - User Ratings']))/(max(normalizedDF['audience - User Ratings']) - min(normalizedDF['audience - User Ratings']))
    #rt.update(normalized1)
    #normalized1 = (normalizedDF['Reviews Counted']- min(normalizedDF['Reviews Counted']))/(max(normalizedDF['Reviews Counted']) - min(normalizedDF['Reviews Counted']))
    #rt.update(normalized1)
    #normalized1 = (normalizedDF['Average Rating']- min(normalizedDF['Average Rating']))/(max(normalizedDF['Average Rating']) - min(normalizedDF['Average Rating']))
    #rt.update(normalized1)
    #normalized1 = (normalizedDF['Rotten']- min(normalizedDF['Rotten']))/(max(normalizedDF['Rotten']) - min(normalizedDF['Rotten']))
    #rt.update(normalized1)
    #normalized1 = (normalizedDF['audience - Average Rating']- min(normalizedDF['audience - Average Rating']))/(max(normalizedDF['audience - Average Rating']) - min(normalizedDF['audience - Average Rating']))
    #rt.update(normalized1)
    normalized1 = (normalizedDF['Box Office']- min(normalizedDF['Box Office']))/(max(normalizedDF['Box Office']) - min(normalizedDF['Box Office']))
    rt.update(normalized1)
    return rt
    
def performanceForStudio(rt):
    for index,row in rt.iterrows():
        studioName = row['Studio']
        df2 = rt.loc[rt['Studio'] == studioName]
        df2 = df2.sort_values('year', ascending=False)
        df2 = df2.reset_index()
        if len(df2) == 0:
                rt.at[index,'studio_performance'] = 0
        elif not ((df2.loc[df2['movie_id'] == row['movie_id']]).empty):
            index_value = (df2.loc[df2['movie_id'] == row['movie_id']].index[0])
            if (index_value == len(df2)-1):
               # sum_value = (row['length']+row['Box Office'])/2#+row['Rotten']+row['Reviews Counted']+row['audience - Average Rating']+row['audience - User Ratings']+row['Average Rating']+row['Box Office'])/8
                rt.at[index,'studio_performance'] = 0
            else:
                dfx = df2[index_value+1: index_value+6]
                sum_value = (dfx['length']+dfx['Box Office'])/2#+dfx['Rotten']+dfx['Reviews Counted']+dfx['audience - Average Rating']+dfx['audience - User Ratings']+dfx['Average Rating']+dfx['Box Office'])/8
                meanValue = np.nanmean(sum_value)
                rt.at[index,'studio_performance'] = meanValue
    return rt
    
def getPerformanceForEachActor(rt,row, index, actorName,i):
    df2 = rt.loc[(rt['actor1'] == str(actorName).strip()) | (rt['actor2'] == str(actorName).strip()) | (rt['actor3'] == str(actorName).strip())] #| (rt['actor4'] == str(actorName).strip()) | (rt['actor5'] == str(actorName).strip()) ]
    df2 = df2.sort_values('year',ascending = False)
    df2 = df2.reset_index()
    sum_value = 0
    if len(df2) == 0:
        rt.at[index,'actor'+str(i)+'Performance'] = 0
    elif not ((df2.loc[df2['movie_id'] == row['movie_id']]).empty):
        index_value = (df2.loc[df2['movie_id'] == row['movie_id']].index[0])
        if (index_value == len(df2)-1):
            #sum_value = (row['length']+row['Fresh']+row['Rotten']+row['Reviews Counted']+row['audience - Average Rating']+row['audience - User Ratings']+row['Average Rating']+row['Box Office'])/8
            rt.at[index,'actor'+str(i)+'Performance'] = 0
        else:
            dfx = df2[index_value+1 : index_value+6]
            sum_value = (dfx['length']+dfx['Box Office'])/2#+dfx['Rotten']+dfx['Reviews Counted']+dfx['audience - Average Rating']+dfx['audience - User Ratings']+dfx['Average Rating']+dfx['Box Office'])/8
            meanValue = np.nanmean(sum_value)        
            rt.at[index,'actor'+str(i)+'Performance'] = meanValue
    return rt
    
    
def performanceForActor(rt):
    for index, row in rt.iterrows():
        actor1 = str(row['actor1']).strip()
        actor2 = str(row['actor2']).strip()
        actor3 = str(row['actor3']).strip()
        #actor4 = str(row['actor4']).strip()
        #actor5 = str(row['actor5']).strip()
        list_actors = [actor1,actor2,actor3]#actor4,actor5]
        i=1
        for actor in list_actors:
            if actor:
                rt = getPerformanceForEachActor(rt,row,index,actor,i)
            else: 
                rt.at[index,'actor'+str(i)+'Performance'] = 0
            i+=1
    return rt
    
def getPerformanceForEachDirector(rt, row, index, directorName):
    df2 = rt.loc[(rt['director1'] == str(directorName).strip())] #| (rt['director2'] == str(directorName).strip()) ]
    df2 = df2.sort_values('year',ascending = False)
    df2 = df2.reset_index()
    sum_value = 0
    if len(df2) == 0:
        rt.at[index,'directorPerformance'] = 0
    elif not ((df2.loc[df2['movie_id'] == row['movie_id']]).empty):
        index_value = (df2.loc[df2['movie_id'] == row['movie_id']].index[0])
        if (index_value == len(df2)-1):
            #sum_value = (row['length']+row['Fresh']+row['Rotten']+row['Reviews Counted']+row['audience - Average Rating']+row['audience - User Ratings']+row['Average Rating']+row['Box Office'])/8
            rt.at[index,'directorPerformance'] = 0
        else:
            dfx = df2[index_value+1 : index_value+6]
            sum_value = (dfx['length']+dfx['Box Office'])#+dfx['Rotten']+dfx['Reviews Counted']+dfx['audience - Average Rating']+dfx['audience - User Ratings']+dfx['Average Rating']+dfx['Box Office'])/8
            meanValue = 0
            meanValue = np.nanmean(sum_value)        
            rt.at[index,'directorPerformance'] = meanValue
    return rt
    
def performanceForDirector(rt):
    for index, row in rt.iterrows():
        director1 = row['director1']
        #director2 = row['director2']
        list_director = [director1]#, director2]
        for director in list_director:
            if director:
                 rt = getPerformanceForEachDirector(rt,row,index,director)
            else:
                rt.at[index,'directorPerformance'] = 0
    return rt
            

rt=pd.read_csv('rotten_impute.csv',encoding='utf-8')
#rt = rt.drop_duplicates(subset=['movie_id'], keep = False)
"""droplist = ['Unnamed: 0','Unnamed: 0.2','Unnamed: 0.1','Unnamed: 0.1.1','director1.1','director2.1','In Theaters', 'Runtime','criticConsensus','Genre','synopsis','criticReviews','day',
            'genre','topAudienceReviews','On Disc/Streaming','actor4','actor5', 'writer1','writer2','director2','Rating', 
            'actor4_star','actor4_oscars','actor4_nominations','actor4_otherWins','actor4_bignominations',
           'actor5_star','actor5_oscars','actor5_nominations','actor5_otherWins','actor5_bignominations',
           'director2_star','director2_oscars','director2_nominations','director2_otherWins','director2_bignominations']
#rt = rt.drop(droplist,axis=1)"""
rt = scaleTheData(rt)
rt = performanceForStudio(rt)

#tr = rt
#actords = pd.read_csv('imdb_awd.csv',encoding="utf-8")
#actords = actords.drop_duplicates(subset=['movie_id'], keep = False)
#actords = pd.concat((katre['movie_id'],actords),axis=1)
#tr.set_index('movie_id', inplace=True)
#actords.set_index('movie_id', inplace=True)
#maj = tr.join(actords,how='left')
maj = rt
maj['actor1'] = maj['actor1'].replace("'","").str.strip()
maj['actor2'] = maj['actor2'].replace("'","").str.strip()
maj['actor3'] = maj['actor3'].replace("'","").str.strip()
#maj['actor4'] = maj['actor4'].replace("'","").str.strip()
#maj['actor5'] = maj['actor5'].replace("'","").str.strip()
maj['actor1'] = maj['actor1'].replace('"',"").str.strip()
maj['actor2'] = maj['actor2'].replace('"',"").str.strip()
maj['actor3'] = maj['actor3'].replace('"',"").str.strip()
#maj['actor4'] = maj['actor4'].replace('"',"").str.strip()
#maj['actor5'] = maj['actor5'].replace('"',"").str.strip()
maj['director1'] = maj['director1'].replace('"',"").str.strip()
maj['director1'] = maj['director1'].replace("'","").str.strip()
#maj['director2'] = maj['director2'].replace('"',"").str.strip()
#maj['director2'] = maj['director2'].replace("'","").str.strip()
#maj['movie_id'] = maj.index
#maj = maj.reset_index(drop=True)
mapd = performanceForActor(maj)
map1 = performanceForDirector(maj)
"""dummy=['Wide Release']
final_data = pd.get_dummies(map1,columns=dummy,drop_first=True)
final_data.rename(columns={'Unnamed: 0':'id2'}, inplace=True)
droplist = ['Studio','audience - User Ratings','tomatoMeter','Reviews Counted','audience - Average Rating','Rotten',
            'Fresh','Rotten','actor1','actor2','actor3','director1','movie_id','Fresh','Box Office','Average Rating','Reviews Counted','criticRating',
            'audienceMeter','movie_id','ratings']
final_data.to_csv('FinalVer1.csv',encoding='utf-8')

workingdata = final_data.drop(droplist,axis=1)"""
map1.to_csv('UpdatedPerformance.csv',encoding='utf-8')

## Imputation
"""
droplist = ['id','id2','Studio','audience - User Ratings','Reviews Counted','audience - Average Rating','Rotten',
            'tomatoMeter_fresh','tomatoMeter_rotten','actor1','actor2','actor3','actor4','actor5','director1',
            'director2','movie_id','writer1','writer2','Fresh','Box Office','Average Rating']
map1.rename(columns={'Unnamed: 0':'id2'}, inplace=True)
map2 = map1.drop(droplist,axis=1)
map2['director2Performance'] = map2['director2Performance'].replace(0,np.nan)
map2['director1Performance'] = map2['director1Performance'].replace(0,np.nan)
map2['actor1Performance'] = map2['actor1Performance'].replace(0,np.nan)
map2['actor2Performance'] = map2['actor2Performance'].replace(0,np.nan)
map2['actor3Performance'] = map2['actor3Performance'].replace(0,np.nan)
map2['actor4Performance'] = map2['actor4Performance'].replace(0,np.nan)
map2['actor5Performance'] = map2['actor5Performance'].replace(0,np.nan)

nans = lambda df: df[df.isnull().any(axis=1)]
X_incomplete=nans(map2)
mapk=pd.DataFrame(data=KNN(k=3).complete(X_incomplete), columns=X_incomplete.columns, 
                 index=X_incomplete.index)
map2.update(mapk)

X = map2.drop('diff_rating',axis=1)
y = map2['diff_rating']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)



lm = LinearRegression()
lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)
print("R2: ", lm.score)
predictions = lm.predict( X_test)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients

r2_score(y_test, predictions)
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, predictions, color='blue', linewidth=3)"""