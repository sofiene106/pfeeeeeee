#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import chi2_contingency
from collections import Counter
from math import sqrt


# In[2]:


df = pd.read_csv(r'C:\Users\sofi\Downloads\wetransfer-3ab4ca\2018.csv',sep=';', encoding = 'ISO-8859-1',
                     names=['ID_ORDER', 'DATE-ADD','TIME-ADD','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE'],
                     dtype={'ID_ORDER':int,'DATE-ADD':object,'TIME-ADD':object,'LOCATION':int,'ID_CUSTOMER':object,'LAST_NAME':object,'FIRST_NAME':object,'YX_LIBELLE':object,'BIRTH_YEAR':object,'TELEX':object,'EMAIL':object,'ADRESS':object,'POSTAL_CODE':object,'CITY':object,'ITEM_CODE':object,'CC_LIBELLE':object,'CC_LIBELLE_1':object})

df


# In[3]:


df.isnull().sum()


# In[4]:


df['PVTTC'].value_counts()


# In[5]:


df['PVTTC'] = df['PVTTC'].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float).dropna()


# In[6]:


df[df['PVTTC']==0]


# In[7]:


df = df[df.PVTTC != 0]


# In[8]:


df


# In[9]:


df.info()


# In[10]:


df.replace({'0000-00-00': np.nan},inplace=True)


# In[11]:


df.isnull().sum()


# In[12]:


#convert DataFrame column to date-time:'GP_DATEPIECE'
df['DATE-ADD'] = pd.to_datetime(df['DATE-ADD'], format = '%Y-%m-%d')
df['DATE-ADD']= pd.to_datetime(df['DATE-ADD'], errors='ignore')
df.head()


# In[13]:


df['TIME-ADD'] = pd.to_datetime(df['TIME-ADD'])

df


# In[14]:


# Create the dictionary for  ETABLISSEMENT
etab = {28:'Sasio Geant',31:'Blue Island Carrefour',62:'Central Park',16:'Sousse',8:'Lafayette',130:'Blue Island Palmarium',50:'Sasio Carrefour',24:'Bizerte',40:'Ennasr',56:'Sasio Manzah VI',14:'Blue Island Zephyr',61:'La Soukra',63:'Sasio Menzah V',54:'Nabeul',37:'Sasio Zephyr',134:'Sasio Palmarium',64:'Nabeul',65:'Blue Island Manar',36:'Sasio Manar',25:'Blue Island Djerba',52:'Blue Island Menzah VI',66:'Mehdia',42:'Lac 2',67:'Sfax',68:'Monastir',51:'Blue Island Menzah V',69:'El Kef',35:'Kairouan',15:'Sasio Mseken',27:'Sasio Mseken',60:'Sasio Djerba',18:'Kelibia',41:'Ksar Hellal',74:'Hammamet'}
df['LOCATION'] = df['LOCATION'].map(etab)
#display the first 5 lines
df


# In[15]:


df.isnull().sum()


# In[16]:


df['DATE-ADD'] = pd.to_datetime(df['DATE-ADD'])

df['SEASON'] = (df['DATE-ADD'].dt.month - 1) // 3
df['SEASON'] += (df['DATE-ADD'].dt.month == 3)&(df['DATE-ADD'].dt.day>=20)
df['SEASON'] += (df['DATE-ADD'].dt.month == 6)&(df['DATE-ADD'].dt.day>=21)
df['SEASON'] += (df['DATE-ADD'].dt.month == 9)&(df['DATE-ADD'].dt.day>=23)
df['SEASON'] -= 3*((df['DATE-ADD'].dt.month == 12)&(df['DATE-ADD'].dt.day>=21)).astype(int)


# In[17]:


season={0:'Winter',1:'Spring',2:'Summer',3:'Autumn'}

df['SEASON'] = df['SEASON'].map(season)
df


# In[18]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[19]:


df['COVID']='Pre-Covid'
df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','COVID','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[20]:


df['BIRTH_YEAR'].value_counts()


# In[21]:


df['BIRTH_YEAR'] = pd.to_datetime(df['BIRTH_YEAR'],errors='coerce')
df.isnull().sum()


# In[22]:


df['BIRTH_YEAR'].value_counts()


# In[23]:


df.BIRTH_YEAR[df.BIRTH_YEAR.dt.year > 2018] = np.nan
df.BIRTH_YEAR[df.BIRTH_YEAR.dt.year < 1918] = np.nan
df.isnull().sum()


# In[ ]:





# In[ ]:





# In[24]:


df['BIRTH_YEAR'] = pd.to_datetime(df['BIRTH_YEAR'],errors='coerce')


# In[25]:


df['AGE'] = df['DATE-ADD'].dt.year- df['BIRTH_YEAR'].dt.year


# In[26]:


df['AGE'].value_counts()


# In[ ]:





# In[27]:


df['AGE'].max()


# In[28]:


df[df['AGE']==99]


# In[29]:


df['AGE'].min()


# In[30]:


max_AGE=df['AGE'].max()
min_AGE=df['AGE'].min()


# In[31]:


cut_age = ['Moins de 25 ans','entre 25 et 30 ans', 'entre 30 et 35 ans','entre 35 et 40 ans','entre 40 et 50 ans','Plus de 50 ans']
cut_bins =[min_AGE, 25, 30, 35,40,50, max_AGE]
df['AGE_SEGMENT'] = pd.cut(df['AGE'], bins=cut_bins, labels = cut_age)


# In[32]:


df['AGE_SEGMENT'].value_counts()


# In[33]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','COVID','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','AGE_SEGMENT','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[34]:


df['CONFINEMENT']='NON'
df['CURFEW']='NON'
df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','COVID','CONFINEMENT','CURFEW','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]
df


# In[35]:


df['P_MONTH']=0
df['P_MONTH']+=(df['DATE-ADD'].dt.day>10)
df['P_MONTH']+=(df['DATE-ADD'].dt.day>20)
df['P_MONTH']=df['P_MONTH'].astype(int)


# In[36]:


month={0:'Start_of_Month',1:'Middle_of_Month',2:'End_of_Month'}
df['P_MONTH'] = df['P_MONTH'].map(month)


# In[37]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','P_MONTH','COVID','CONFINEMENT','CURFEW','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]
df


# In[38]:


from datetime import date
import calendar


df['P_WEEK']=df['DATE-ADD'].dt.strftime('%A')
week={'Monday':'Start_of_Week','Tuesday':'Start_of_Week','Wednesday':'Mid_Week','Thursday':'Mid_Week','Friday':'Mid_Week','Saturday':'Week_End','Sunday':'Week_end'}
df['P_WEEK'] = df['P_WEEK'].map(week)


# In[39]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','P_MONTH','P_WEEK','COVID','CONFINEMENT','CURFEW','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[40]:


df['QTEFACT'].value_counts()


# In[41]:


df = df[df.QTEFACT != 98.9]


# In[42]:


df


# In[43]:


df=df[df.QTEFACT!=0.0000]


# In[44]:


df


# In[45]:


df['QTEFACT'].value_counts()


# In[46]:


df['QTEFACT']=df['QTEFACT'].astype(float)


# In[47]:


df['TOTAL_QUANTITY'] = df['QTEFACT'].groupby(df['ID_ORDER']).transform('sum')


# In[48]:


df


# In[49]:


df[df['ID_ORDER']==66032518]


# In[50]:


df['TOTAL_QUANTITY'].value_counts()


# In[51]:


max_QTE=df['TOTAL_QUANTITY'].max()
min_QTE=df['TOTAL_QUANTITY'].min()


# In[52]:


cut_qte = ['1 pièce', "2 pièces", 'Entre 3 et 5 Pièces', 'Entre 6 et 10 Pièces', 'Plus que 10 Pièces']
cut_bins =[1,1.5, 2, 5, 10, max_QTE ]
df['TOTAL_QUANTITY_SEGMENTS'] = pd.cut(df['TOTAL_QUANTITY'], bins=cut_bins, labels = cut_qte)


# In[53]:


df['TOTAL_QUANTITY_SEGMENTS'].value_counts()


# In[ ]:





# In[54]:


df2= pd.read_csv(r'C:\Users\sofi\Downloads\train_code_postal.csv')
df2 = df2.rename(columns={'code postal': 'POST_CODE'})
df2.info()


# In[55]:


hour=df['TIME-ADD'].dt.hour
hour


# In[56]:


# Create the dictionary for Hour 

H = {0:'Early_morning', 1:'Early_morning', 2:'Early_morning', 3:'Early_morning',4:'Early_morning',5:'Early_morning',6:'Early_morning',7:'Early_morning',8:'Early_morning',9:'Late_morning',10:'Late_morning',11:'Late_morning',12:'Early_afternoon',13:'Early_afternoon',14:'Early_afternoon',15:'Late_afternoon',16:'Late_afternoon',17:'Late_afternoon',18:'Evening',19:'Evening',20:'Evening',21:'Night',22:'Night',23:'Night'}
# Use the dictionary to map the 'Hour'
df['HOUR'] = hour.map(H)
#display the first 5 lines
df


# In[57]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','HOUR','SEASON','P_MONTH','P_WEEK','COVID','CONFINEMENT','CURFEW','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE','TOTAL_QUANTITY']]
df


# In[ ]:





# In[58]:


new_row2070 = {'POST_CODE':'2070', 'Delegation':'LA MARSA', 'poverty rate':2.2, 'zone':'urbaine','orientation':'nord-est'}
df2 = df2.append(new_row2070, ignore_index=True)


# In[59]:


df['POSTAL_CODE']=df['POSTAL_CODE'].apply(str)
df2['POST_CODE']=df2['POST_CODE'].apply(str)
df2.info()


# In[60]:


df2


# In[61]:


DF10=pd.DataFrame()
DF10['codep']=df2['POST_CODE']
DF10

    
        
    
    
    
    
    

    
    
     
        


# In[62]:


def tunis(code,train):
 try:
    if str(code)[:2] == '20':
        return 'Grand-Tunis'
    elif str(code)[:2] == '10':
        return 'Grand-Tunis'
    elif str(code)[:2] == '11':
        return 'Grand-Tunis'
    else :
        return str(train[train['POST_CODE']== str(code)]['orientation'].values[0])
 except Exception :
    print (str(code))

            
            


# In[63]:


DF10['ORIENTATION']=DF10.apply(lambda x: tunis(x.codep,df2), axis=1)


# In[64]:


DF10


# In[65]:


def get_region(code, train):
  try:
    if str(code) in str(train['codep'].unique()):
        return str(train[train['codep']== str(code)]['ORIENTATION'].values[0])
  except Exception :
    print (str(code))


# In[66]:


df['region'] = df.apply(lambda x: get_region(x.POSTAL_CODE, DF10), axis=1)


# In[67]:


df['region'].value_counts()


# In[68]:


df['POSTAL_CODE'][df['region'].isnull()]


# In[69]:


df[df['POSTAL_CODE']=='2070']


# In[70]:


def get_area(code, train):
  try:
    if str(code) in str(train['POST_CODE'].unique()):
        return str(train[train['POST_CODE']== str(code)]['zone'].values[0])
  except Exception :
    print (str(code))


# In[71]:


df['area'] = df.apply(lambda x: get_area(x.POSTAL_CODE, df2), axis=1)


# In[72]:


df = df.rename(columns={'region': 'REGION'})
df = df.rename(columns={'area': 'AREA'})


# In[73]:


df['AREA'].value_counts()


# In[74]:


cut_class = ['A', 'B', 'C+','C-', 'D', 'E']
cut_bins =[0, 2, 8, 18, 26, 42, 55]
df2['class'] = pd.cut(df2['poverty rate'], bins=cut_bins, labels = cut_class)
df2


# In[ ]:





# In[ ]:





# In[75]:


def get_class22(code, train):
  try:
    if str(code) in str(train['POST_CODE'].unique()):
      return str(train[train['POST_CODE']== str(code)]['class'].values[0])
  except Exception :
    print (str(code))


# In[76]:


df['CLASS'] = df.apply(lambda x: get_class22(x.POSTAL_CODE, df2), axis=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[77]:


df['CLASS'].value_counts()


# In[ ]:





# In[78]:


df.isnull().sum()


# In[79]:


df.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[80]:


df.to_csv(r'C:\Users\sofi\Desktop\Data.csv', index = False)


# In[81]:


df[df['TIME-ADD'].isnull()]


# In[ ]:





# In[ ]:





# In[82]:


df = df.rename(columns={'BIRTH_YEAR': 'BIRTHDAY'})
df = df.rename(columns={'ITEM_CODE': 'PRODUCT_ID'})
df = df.rename(columns={'CC_LIBELLE_1': 'PRODUCT_NAME'})
df = df.rename(columns={'CC_LIBELLE': 'LABEL'})
df = df.rename(columns={'DESIGNATION': 'COLOR'})
df = df.rename(columns={'PVTTC': 'PRODUCT_PRICE'})
df = df.rename(columns={'PUTTCNET': 'PRODUCT_PRICE_AFTER_REDUCTION'})
df = df.rename(columns={'QTEFACT': 'PRODUCT_QUANTITY'})
df = df.rename(columns={'MLR_REMISE': 'REDUCTION_PERCENT'})
df = df.rename(columns={'GTR_LIBELLE': 'REDUCTION_TYPE'})
df = df.rename(columns={'YX_LIBELLE': 'CUSTOMER_DESCRIPTION'})





# In[83]:


data= pd.read_csv(r'C:\Users\sofi\Downloads\train_gender.csv')
data.info()


# In[84]:


df.isnull().sum()


# In[85]:


df


# In[86]:


df['REDUCTION_PERCENT'].value_counts()


# In[87]:


max_red=df['REDUCTION_PERCENT'].max()
min_red=df['REDUCTION_PERCENT'].min()


# In[88]:


min_red


# In[89]:


max_red


# In[90]:


cut_qte = ['Sans remise', "Inférieur à 30%", 'de 30 à 40%', 'de 40 à 50%','Plus que 50%' ]
cut_bins =[-0.000000001,0.000000001, 30, 40, 50, max_red ]
df['REDUCTION_PERCENT_SEGMENTS'] = pd.cut(df['REDUCTION_PERCENT'], bins=cut_bins, labels = cut_qte)


# In[91]:


df.isnull().sum()


# In[92]:


df['REDUCTION_PERCENT_SEGMENTS'].value_counts()


# In[93]:


df[df['REDUCTION_PERCENT_SEGMENTS']=='Sans remise']


# In[94]:


df.info()


# In[95]:


df['TOTAL_BASKET']=df['PRODUCT_PRICE_AFTER_REDUCTION'].astype(float)*df['PRODUCT_QUANTITY']


# In[96]:


df.isnull().sum()


# In[97]:


df['TOTAL_BASKET'].value_counts()


# In[98]:


df['TOTAL_TICKET']=df['TOTAL_BASKET'].groupby(df['ID_ORDER']).transform('sum')


# In[99]:


df.isnull().sum()


# In[100]:


df['TOTAL_TICKET'].value_counts()


# In[101]:


max_v=df['TOTAL_TICKET'].max()
min_v=df['TOTAL_TICKET'].min()


# In[102]:


cut_ticket = ['Inférieur à 100 DT', 'Entre 100 et 200 DT', 'Entre 200 et 300 DT','Supérieur à 300 DT']
cut_bins =[min_v, 100,200,300,max_v ]
df['TICKET_VALUE'] = pd.cut(df['TOTAL_TICKET'], bins=cut_bins, labels = cut_ticket)


# In[103]:


df['TICKET_VALUE'].value_counts()


# In[120]:


df_g= pd.read_csv(r'C:\Users\sofi\Downloads\train_gender.csv')
df_g.info()


# In[121]:


def word2vec(word):
    # Count the number of characters in each word.
    count_characters = Counter(word)
    # Gets the set of characters and calculates the "length" of the vector.
    set_characters = set(count_characters)
    length = sqrt(sum(c*c for c in count_characters.values()))
    return count_characters, set_characters, length, word


# In[122]:


def cosine_similarity(vector1, vector2, ndigits):
    # Get the common characters between the two character sets
    common_characters = vector1[1].intersection(vector2[1])
    # Sum of the product of each intersection character.
    product_summation = sum(vector1[0][character] * vector2[0][character] for character in common_characters)
    # Gets the length of each vector from the word2vec output.
    length = vector1[2] * vector2[2]
    # Calculates cosine similarity and rounds the value to ndigits decimal places.
    if length == 0:
        # Set value to 0 if word is empty.
        similarity = 0
    else:
        similarity = round(product_summation/length, ndigits)
    return similarity


# In[123]:


vector_list = [word2vec(str(i)) for i in df_g['name'].unique()]


# In[124]:


vector_list


# In[125]:



def find_similar(vector_list,request_word,threshold):
    # Initiate an empty list to store results.
    results_list = []
    # Two loops to compare each vector with another vector only once.
    vector2= word2vec(str(request_word))
    max_similarity = 0 
    similar_word = None
    for i in range(len(vector_list)):
        vector1 = vector_list[i]
        # Calculate cosine similarity
        similarity_score = cosine_similarity(vector1, vector2, 3)
        # Append to results list if similarity score is between 1 and the threshold.
        # Note that scores of 1 can be ignored here if we want to exclude people with the same name.
        if similarity_score> threshold :
            if 1 >= similarity_score>=  max_similarity:
                max_similarity = similarity_score
                similar_word = vector1
    if max_similarity>0 :
        return similar_word[3], max_similarity
    return None,0


# In[126]:



#generating genders from firstnames and lastnames

def get_gender(firstname,lastname,train):
    similar_firstname , similarity_score_firstname = find_similar(vector_list,firstname,0.5)
    similar_lastname , similarity_score_lastname = find_similar(vector_list,lastname,0.5)
    if lastname =='CLIENT de PASSAGE' :
        return None
    if similarity_score_firstname > 0 :
        
        gender_firstname=str(train[train['name']== str(similar_firstname)]["gender 1"].values[0])
        if similarity_score_lastname> 0 :
            
            gender_lastname= str(train[train['name']== str(similar_lastname)]["gender 1"].values[0])
            if gender_firstname!=gender_lastname : 
                if similarity_score_firstname>= similarity_score_lastname :
                    return gender_firstname
                else :
                    return gender_lastname
            else:
                return gender_firstname
        elif similarity_score_lastname> 0:
            gender_lastname= str(train[train['name']== str(similar_lastname)]["gender 1"].values[0])
            return gender_lastname

      


# In[127]:


#defining the scores of each name
def get_score(firstname,lastname, train):
  try:
    similar_firstname , similarity_score_firstname = find_similar(vector_list,firstname,0.5)
    similar_lastname , similarity_score_lastname = find_similar(vector_list,lastname,0.5)
    if lastname =='CLIENT de PASSAGE' :
        return 0
    if similarity_score_firstname > 0 :
      if similarity_score_lastname> 0 :
        return max(similarity_score_firstname,similarity_score_lastname)
    elif similarity_score_lastname> 0:
      return similarity_score_lastname
    else :
      return 0.5
  except Exception as e :
    print (e)


# In[112]:


df["FIRST_NAME"]=df["FIRST_NAME"].apply(str)
df["LAST_NAME"]=df["LAST_NAME"].apply(str)


# In[113]:


df = df.rename(columns={'FIRST_NAME': 'firstname'})
df = df.rename(columns={'LAST_NAME': 'lastname'})


# In[114]:


df.head()


# In[ ]:


df['gender'] = df.apply(lambda x: get_gender(x.firstname, x.lastname,df_g), axis=1)
df["score"]=df.apply( lambda x: get_score(x.firstname, x.lastname,df_g), axis=1)


# In[116]:


df['gender'].value_counts()


# In[117]:


df['firstname'][df['gender']=='10']


# In[119]:


df.isnull().sum()

