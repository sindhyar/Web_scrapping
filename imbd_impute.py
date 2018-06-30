#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 22:24:05 2018

@author: SaiSanthosh
"""

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


def finalmerge_actors(name_a,rank_a,oscars_a,nominations_a,another_wins_a,bignominations_a):

    x = -1
    #name_a = set(name_a)
    imdb_actor1 = pd.DataFrame(columns=['actor1','actor1_star', 'actor1_oscars','actor1_nominations','actor1_otherWins','actor1_bignominations'])
    imdb_actor2 = pd.DataFrame(columns=['actor2','actor2_star', 'actor2_oscars','actor2_nominations','actor2_otherWins','actor2_bignominations'])
    imdb_actor3 = pd.DataFrame(columns=['actor3','actor3_star', 'actor3_oscars','actor3_nominations','actor3_otherWins','actor3_bignominations'])
    
    for i in imdb_new.loc[:,'actor1']:
        i=str(i).strip()
        
        x=x+1
        a= -1
        for j in name_a:
            j=str(j).strip()
            
            a=a+1
            if i == j:
                #print(i)
                #print(j)
                #x=x+1
                imdb_actor1.loc[x,'actor1'] = i
                imdb_actor1.loc[x,'actor1_star'] = rank_a[a]
                imdb_actor1.loc[x,'actor1_oscars'] = oscars_a[a]
                imdb_actor1.loc[x,'actor1_nominations'] = nominations_a[a]
                imdb_actor1.loc[x,'actor1_otherWins'] = another_wins_a[a]
                imdb_actor1.loc[x,'actor1_bignominations'] = bignominations_a[a]
    
         
    y= -1
    for i in imdb_new.loc[:,'actor2']:
        i=str(i).strip()
        y=y+1
        b=-1
        for j in name_a:
            j=str(j).strip()
            b = b+1
            if i == j:
                imdb_actor2.loc[y,'actor2'] = i
                imdb_actor2.loc[y,'actor2_star'] = rank_a[b]
                imdb_actor2.loc[y,'actor2_oscars'] = oscars_a[b]
                imdb_actor2.loc[y,'actor2_nominations'] = nominations_a[b]
                imdb_actor2.loc[y,'actor2_otherWins'] = another_wins_a[b]
                imdb_actor2.loc[y,'actor2_bignominations'] = bignominations_a[b]
                
                
    z= -1
    
    for i in imdb_new.loc[:,'actor3']:
        i=str(i).strip()
        z=z+1
        c=-1
        for j in name_a:
            j=str(j).strip()
            c = c+1
            if i == j:
                imdb_actor3.loc[z,'actor3'] = i
                imdb_actor3.loc[z,'actor3_star'] = rank_a[c]
                imdb_actor3.loc[z,'actor3_oscars'] = oscars_a[c]
                imdb_actor3.loc[z,'actor3_nominations'] = nominations_a[c]
                imdb_actor3.loc[z,'actor3_otherWins'] = another_wins_a[c]
                imdb_actor3.loc[z,'actor3_bignominations'] = bignominations_a[c]
                
                
    
    imdb_actor = pd.concat([imdb_actor1,imdb_actor2,imdb_actor3], axis=1)
         
    return imdb_actor


###merging directors information to their original format
def finalmerge_directors(name_d,rank_d,oscars_d,nominations_d,another_wins_d,bignominations_d):

    x = -1 
    imdb_director1 = pd.DataFrame()
    imdb_director1 = pd.DataFrame(columns=['director1','director2',
                                        'director1_star', 'director1_oscars','director1_nominations','director1_otherWins','director1_bignominations',
                                        'director2_star', 'director2_oscars','director2_nominations','director2_otherWins','director2_bignominations'])
    
    for i in imdb_new.loc[:,'director1']:
        i=str(i).strip()
        x=x+1
        a= -1
        for j in name_d:
            j=str(j).strip()
            a=a+1
            if i == j:
                #x=x+1
                imdb_director1.loc[x,'director1'] = i
                imdb_director1.loc[x,'director1_star'] = rank_d[a]
                imdb_director1.loc[x,'director1_oscars'] = oscars_d[a]
                imdb_director1.loc[x,'director1_nominations'] = nominations_d[a]
                imdb_director1.loc[x,'director1_otherWins'] = another_wins_d[a]
                imdb_director1.loc[x,'director1_bignominations'] = bignominations_d[a]
            

     
    
               
    return imdb_director1

if __name__ == "__main__":
    
    imdb_new = pd.read_csv('imdb_awd.csv',encoding='utf-8')
    imdb_new.replace(r'\s+', np.nan, regex=True)
    
    
    
    ###Run 5th- reading data from out_* files into three dfs and outputs single final_imdb.csv file. Merges for all the rows.
    #######Should run in one go
    
    out_actor = pd.read_csv('test_actor.txt',sep='\t',header=None, encoding='utf-8')
    out_actor.columns = ['index','name_a', 'rank_a', 'oscars_a', 'nominations_a', 'another_wins_a', 'bignominations_a']
    out_actor.loc[:,'name_a'] = out_actor.loc[:,'name_a'].str.replace('"','')
    out_actor.loc[:,'name_a'] = out_actor.loc[:,'name_a'].str.replace("'","")
    #out_writer = pd.read_csv('out_writer.csv')
    out_director = pd.read_csv('test_director.txt',sep='\t',header=None)
    out_director.columns = ['index','name_d', 'rank_d', 'nominations_d', 'another_wins_d', 'bignominations_d']
    oscars_d = []
    for i, v in out_director.loc[:,'rank_d'].iteritems():
        oscars_d.append(repr(v)[-2])
    
    oscars_d_df = pd.DataFrame(oscars_d)
    out_director = pd.concat([out_director, oscars_d_df],axis=1)
    out_director.columns = ['index','name_d', 'rank_d', 'nominations_d', 'another_wins_d', 'bignominations_d','oscars_d']
    
    
    imdb_actors = finalmerge_actors(out_actor.loc[:,'name_a'], out_actor.loc[:,'rank_a'],out_actor.loc[:,'oscars_a']
                        , out_actor.loc[:,'nominations_a'], out_actor.loc[:,'another_wins_a'], out_actor.loc[:,'bignominations_a'])
    
    
    imdb_directors = finalmerge_directors(out_director.loc[:,'name_d'], out_director.loc[:,'rank_d'],out_director.loc[:,'oscars_d']
                       , out_director.loc[:,'nominations_d'], out_director.loc[:,'another_wins_d'], out_director.loc[:,'bignominations_d'])
    
    
    for i in range(len(imdb_new)):
            if i not in imdb_actors.index:
                imdb_actors.loc[i,:] = 'nan'
    imdb_actors.sort_index(inplace=True)
             
            
    for i in range(len(imdb_new)):
            if i not in imdb_directors.index:
                imdb_directors.loc[i,:] = 'nan'
    imdb_directors.sort_index(inplace=True)
    
    final_imdb = pd.concat([imdb_actors, imdb_directors],axis=1)
    
    with open('final_imdb.csv', 'a+') as final:
            final_imdb.to_csv(final, header=True, na_rep = np.nan, encoding='utf-8')
    
    #with open('final_actorimdb.csv', 'a+') as final:
     #       imdb_actors.to_csv(final, header=True, na_rep = np.nan, encoding='utf-8')
    
    rt = pd.read_csv('rotten_org.csv',encoding = 'utf-8')
    rt.replace(r'\s+', np.nan, regex=True)
    
    imdb_actors= imdb_actors.drop(['actor1','actor2','actor3'],axis=1)
    imdb_director = imdb_directors.drop(['director1'], axis=1)
    rt=rt.drop(['actors','Written By','Directed By'],axis=1)
    rt = pd.concat([rt,imdb_new,imdb_actors,imdb_directors], axis = 1)
    rt = rt.drop_duplicates(subset = 'Movie Name', keep = False)
    
    
    with open('imdb_final.csv', 'a+') as final:
            rt.to_csv(final, header=True, na_rep = np.nan, encoding='utf-8')
