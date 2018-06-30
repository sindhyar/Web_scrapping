#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:03:28 2018

@author: SaiSanthosh
"""

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ErrorInResponseException, StaleElementReferenceException
from fake_useragent import UserAgent
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np
import os, re, csv, time

with open('rotten.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split("\t") for line in stripped if line)
    with open('rotten_org.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        #writer.writerow(('movie_id','audience_score','critic_score','actor_names','actor_links','synopsis	In',' Theaters','	Genre','	Studio','Directed By','Runtime','	Box Office','	Rating','Written By'))
        writer.writerows(lines)

rt = pd.read_csv('rotten_org.csv',encoding = 'utf-8')
rt.replace(r'\s+', np.nan, regex=True)
rt.loc[:,'actor_names'] = rt.loc[:,'actor_names'].astype(str)
rt.loc[:,'Written By'] = rt.loc[:,'Written By'].astype(str)
rt.loc[:,'Directed By'] = rt.loc[:,'Directed By'].astype(str)

####Function to divide actors, writers, directors into each column
def namecolumns(actor, writers, directors):
    x =0
    y =0
    z =0
    
    a1 = {}
    a2 = {}
    a3 = {}
    a4 = {}
    a5 = {}
    w1 = {}
    w2= {}
    d1 = {}
    d2 = {}
    for a in actor:
        #print(a)
        print(x)
        #re.sub(r'[^\w]', '', a)
        a=a.strip().split(',')
        try:
            a1[x] = a[0]
           
        except:
            a1[x] = 'nan'
        try:
            a2[x] = a[1]
        except:
            a2[x] = 'nan'
        try:
            a3[x] = a[2]
        except:
            a3[x] = 'nan'
        try:
            a4[x] = a[3]
        except:
            a4[x] = 'nan'
        try:
            a5[x] = a[4]
        except:
            a5[x] = 'nan'
        x=x+1
    
    for w in writers:
        w=w.strip().split(',')
        try:
            w1[y]= w[0]
        except:
            w1[y] = 'nan'
        try:
            w2[y] = w[1]
        except:
            w2[y] = 'nan'
        y=y+1
    
    for d in directors:
        d=d.strip().split(',')
        try:
            d1[z]= d[0]
        except:
            d1[z] = 'nan'
        try:
            d2[z] = d[1]
        except:
            d2[z] = 'nan'
        z=z+1
   
    a1 = pd.DataFrame([a1])
    a2 = pd.DataFrame([a2])
    a3 = pd.DataFrame([a3])
    a4 = pd.DataFrame([a4])
    a5 = pd.DataFrame([a5])
    w1 = pd.DataFrame([w1])
    w2 = pd.DataFrame([w2])
    d1 = pd.DataFrame([d1])
    d2 = pd.DataFrame([d2])
    
    actors1 = pd.concat((a1,a2,a3,a4,a5), axis=0)
    writers1 = pd.concat((w1,w2), axis=0)
    directors1 = pd.concat((d1,d2), axis=0)
    actors=actors1.transpose()
    writers=writers1.transpose()
    directors=directors1.transpose()

    
    return  actors, writers, directors



#******Run 1st- to divide names into each column***********
if __name__ == "__main__":
    
    actors, writers, directors = namecolumns(rt.loc[:,'actor_names'], rt.loc[:,'Written By'], rt.loc[:,'Directed By'])
    consolidate = pd.concat((actors, writers, directors), axis=1)
    consolidate.columns = ['actor1', 'actor2', 'actor3', 'actor4', 'actor5', 'writer1', 'writer2', 'director1', 'director2']
    with open('imdb_awd.csv', 'a+') as imdb:
        consolidate.to_csv(imdb,  encoding='utf-8')
    
