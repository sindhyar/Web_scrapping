#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:23:08 2018

@author: SaiSanthosh
"""

import numpy as np
import pandas as pd
#from fuzzywuzzy import fuzz
#from fuzzywuzzy import process
#import json
#import ast
#from pathlib import Path




def ratingcertificate(rating):
    r=[]
    for x in rating:
        try:
            c=x.split(' ')
            r.append(c[0])
        except:
            r.append(np.nan)
        
    ratings=pd.DataFrame([r])
    #print(ratings)
    return ratings


def runtime(duration):
    r=[]
    for d in duration:
        try:
            time=d.split(' ')
            r.append(time[0])
        except:
            r.append(np.nan)
    length = pd.DataFrame([r])
    #print(length)
    return length
            
        
def releasedate(date):
    d=[]
    m=[]
    y=[]
    r=[]
    for x in date:
        #print(x)
        try:
            release=x.split()
            #print(release)
            #k = release[2].split('¬†')
            y1=release[2]
            m1=release[0]
            d1=release[1]
            r1=release[3]

            d.append(d1)
            m.append(m1)
            y.append(y1)
            r.append(r1)
        except:
            d.append(np.nan)
            m.append(np.nan)
            y.append(np.nan)
            r.append(np.nan)
            
    day = pd.DataFrame([d])
    month = pd.DataFrame([m])
    year = pd.DataFrame([y])
    wide = pd.DataFrame([r])
    
    return day, month, year, wide

# Processing the genre field. Splitting the genre and creating the array of genre elements for each Movie
def processGenre(genre):
    completeGenre = []
    i=0
    for eachMovieGenre in genre:
        i+=1
        movieGenre = []
        try:
            getAllGenre = eachMovieGenre.split(",")
            for getEachAllGenre in getAllGenre:
                movieGenre.append(getEachAllGenre.strip())
            completeGenre.append(movieGenre)
        except:
            completeGenre.append(None)
    return pd.DataFrame([completeGenre])


    
if __name__ == "__main__":

    rt=pd.read_csv('rotten_org.csv',encoding='utf-8')
    
    rt.replace(r'\s+', np.nan, regex=True)
    
    
    day, month, year, wide = releasedate(rt.loc[:,'In Theaters'])
    length=runtime(rt.loc[:,'Runtime'])
    ratings=ratingcertificate(rt.loc[:,'Rating'])
    #Process the genre and return the dataframe object of genre 
    genre = processGenre(rt.loc[:,'Genre'])
    
    day.index.name= 'day'
    month.index.name='month'
    year.index.name='year'
    length.index.name='length'
    wide.index.name = 'release'
    ratings.index.name='ratings'
    
    # Create the new index of the genre in the datafram
    genre.index.name = 'genre'
    
    # Concatinating all the processed data
    df2=pd.concat((day, month,year,wide,length,ratings, genre), axis=0)
    df2 = df2.transpose()
    df2.columns= ['day','month','year','release','length','ratings','genre']
    final_df=pd.concat((rt,df2),axis=1)
    
    t = pd.get_dummies(final_df['genre'].apply(pd.Series).stack()).sum(level=0,axis=0)
    imdb = pd.read_csv('final_imdb.csv',encoding='utf-8')
    final_df = pd.concat([final_df,t, imdb], axis=1)
    
    
    
    for i in range(len(final_df.loc[:,'critic_score'])):
        final_df.loc[i,'critic_score'] = final_df.loc[i,'critic_score'].split('%')[0]
        final_df.loc[i,'audience_score'] = final_df.loc[i,'audience_score'].split('%')[0]
        
    final_df['audience_score']=final_df['audience_score'].astype(float)
    final_df['critic_score']=final_df['critic_score'].astype(float)
        
    final_df['diff_rating']=final_df['critic_score'].sub(final_df['audience_score'], axis='index')
    final_df['diff_rating']=abs(final_df['diff_rating'])
    
    final_df.to_csv('rotten_predata.csv', na_rep = np.nan, encoding='utf-8')
