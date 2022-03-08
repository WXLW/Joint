#!/usr/bin/env python
# coding: utf-8
from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


# Importing data using pandas
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.regression import *
from timeit import default_timer as timer

#data = pd.read_csv('G:/Jupyter/Joint_flexural strength_3.csv')#, encoding= 'unicode_escape')
data = pd.read_csv('Joint_flexural strength_combination_9.csv')#, encoding= 'unicode_escape')


exp_reg101 = setup(data = data, target = 'Mu', session_id=123,train_size = 0.8, 
                  normalize = True,normalize_method='minmax', #transformation = True, transform_target = True, 
                  remove_outliers = True, outliers_threshold = 0.05,
                  silent =True)




X = get_config('X') 
y = get_config('y')
X_train = get_config('X_train') 
y_train = get_config('y_train') 
X_test = get_config('X_test') 
y_test = get_config('y_test') 
seed = get_config('seed') 




####################################
#######    0.Deep forest  ##########
####################################



estimator = 'DF'
ensemble =False
method =None
fold = 10
verbose = True
optimize = 'r2'
n_iter = 1000
#setting turbo parameters
cv = 3


#general dependencies
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
import random
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import pearsonr
#setting numpy seed
np.random.seed(seed)

from deepforest import CascadeForestRegressor
from sklearn.metrics import mean_squared_error
#model = LinearRegression()
full_name = 'Deep forest'


model = CascadeForestRegressor(random_state=seed,n_estimators=2,n_trees=175, max_depth=22)  
model.fit(X.values, y.values)
model11= load_model('Final DF Model 07March2022')


def predict(model,input_df):
    
    prep_pipe_transformer = model11.pop(0)

    Xtest = prep_pipe_transformer.transform(input_df)
    #predictions_df = model.predict(input_df.values)
    y_pred = model.predict(Xtest.values)
    predictions=y_pred[0][0]
    #predictions = predictions_df['Label'][0]
    return predictions


    



def run():
    st.write("""

    # JointDF

    
    """)
    st.write("""

    An App For Predicting Flexural Capacity of Precast Deck Joints in Accelerated Bridge Construction via Deep Ensemble Learning Technique

    
    """)
    #st.write("""

    ## ML-based Practical Bond Strength Prediction App For SRC Structures

    #This app predicts the **Bond strength** between H-steel section and concrete!
 
    #""")
#    #st.write('This App Predicts the **Shear Resistance** of Headed Stud connectors in Steel-Concrete Composite Structures!')
    
    from PIL import Image
    image = Image.open('DF1.png')

    st.image(image, width=1050)#,use_column_width=False)
    
    st.sidebar.header('User Input Features')
    
    Concrete_strength_precast= st.sidebar.slider('Compressive strength of precast concrete, fcm,p (MPa) ', 22, 180, 50)

    Concrete_strength_joint= st.sidebar.slider('Compressive strength of joint concrete, fcm,j (MPa) ', 22, 190, 60)

    Deck_width= st.sidebar.slider('Deck width, b (mm) ', 150, 1800, 700)

    Effective_height= st.sidebar.slider('Effective height, h0 (mm) ', 75, 370, 150)

    Lap_length= st.sidebar.slider('Lap length of longitudinal rebars, l (mm) ', 60, 900, 200)

    Tensile_capacity_longitudinal_rebars= st.sidebar.slider('Tensile capacity of longitudinal rebars, fyAs(kN)', 40, 2900, 400)

    Tensile_capacity_transverse_rebars= st.sidebar.slider('Tensile capacity of transverse rebars, fytAst(kN)', 0, 2500, 130)

    Concrete_type= st.selectbox('Concrete type',['CG','NSC','HSC','SFRC','UHPC'])

    Rebar_type= st.selectbox('Type of rebar connection',[ 'Straight bar', 'Headed bar','U-bar'])

    Interface_type= st.selectbox('Interface type',['Straight', 'Diamond-shaped', 'Curved-shaped', 'T-shaped', 'Notched-shaped', 'Dovetail-shaped'])


    input_dict = {'l': Lap_length,'fcm,p': Concrete_strength_precast, 'fcm,j': Concrete_strength_joint, 
          'b' :Deck_width, 'h0': Effective_height, 'fy*As': Tensile_capacity_longitudinal_rebars,'fyt*Ast': Tensile_capacity_transverse_rebars,
          'Interface type':Interface_type,'Concrete type of joint':Concrete_type, 'Type of rebar':Rebar_type 
                 }
    input_df = pd.DataFrame([input_dict])
    input_df.reset_index(drop=True, inplace=True)    
    
    if st.button("Predict"):
       output = predict(model=model, input_df=input_df)
       output = round(output, 1)
       output =  str(output) +'kN.m'

       st.success('The Flexural Capacity of Precast Deck Joint is  :  {}'.format(output))
    st.info('***Written by Dr. Xianlin Wang,  Department of bridge engineering,  Tongji University,  E-mail:xianlinwang96@gmail.com***')
       
    #output = predict(model=model, input_df=input_df)    
    
    #st.write(output)  assessment of 
        
if __name__ == '__main__':
    run()
