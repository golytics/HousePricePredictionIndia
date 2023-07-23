import pandas as pd
import numpy as np
import pickle
import streamlit as st

#source : https://pypi.org/project/streamlit-analytics/
#import streamlit_analytics

# We use streamlit_analytics to track the site like in Google Analytics
#streamlit_analytics.start_tracking()

# configuring the page and the logo
st.set_page_config(page_title='Mohamed Gabr - House Price Prediction', page_icon ='logo.png', layout = 'wide', initial_sidebar_state = 'auto')


import os
import base64

# the functions to prepare the image to be a hyperlink
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code


# preparing the layout for the top section of the app
# dividing the layout vertically (dividing the first row)
row1_1, row1_2, row1_3 = st.columns((1, 5, 4))

# first row first column
with row1_1:
    gif_html = get_img_with_href('logo.png', 'https://golytics.github.io/')
    st.markdown(gif_html, unsafe_allow_html=True)

with row1_2:
    # st.image('logo.png')
    st.title("Predicting House Prices in Bangalore, India Using Artificial Intelligence")
    st.markdown("<h2>A Machine Learning POC for a Real Estate Client</h2>", unsafe_allow_html=True)

# first row second column
with row1_3:
    st.info(
        """
        ##
        This data product has been prepared as a proof of concept of a machine learning model to predict prices of houses in Cairo, Egypt.
        For demonstration purposes, we have used the data from Bangalore to prove the technical feasibility of the model. Developing the final model required
        many steps following the CRISP-DM methodology. After building the model we used it to predict the prices in this application. **The model can be changed/
        enhanced for any another city based on its own data.**
        """)


from sklearn.ensemble import RandomForestClassifier

st.write("""
        This app predicts the **Prices** of houses in Bangalore
        """)

st.subheader('How to use the model?')
'''
You can use the model by modifying the User Input Parameters on the left. The parameters will be passed to the classification
model and the model will run each time you modify the parameters.

1- You will see the values of the features/ parameters in the **'User Input Parameters'** section in the table below.

2- You will see the prediction result (the price of the house that has the selected parameters in LAKH) under the **'Prediction'** section below.

'''
st.sidebar.header("""User input features/ parameters: 

Select/ modify the combination of features below to predict the price
                """)

# st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
# """)


# get the locations to add them to the listbox
import json
with open("columns.json", "r") as f:
    __data_columns = json.load(f)['data_columns']
    __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk
#print(__locations)


def user_input_features():
    area = st.sidebar.text_input("Area", 1000)
    bhk = st.sidebar.selectbox('BHK (bedroom, hall and kitchen)',(1,2,3,4,5))
    #bath = st.sidebar.selectbox('Bathroom', (1,2,3,4,5))
    bath = st.sidebar.slider('Bathroom', 1,5,3)
    location = st.sidebar.selectbox('Location', (__locations))
    #body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
    data = {'area': area,
            'bhk': bhk,
            'bath': bath,
            'location': location}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Displays the user input features
st.subheader('User Input features')
st.write(input_df)


def get_estimated_price(area,bhk,bath, location):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = area
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1
    # print(x)
    return round(__model.predict([x])[0],2)


# Reads in saved classification model
__model = pickle.load(open('banglore_home_prices_model.pickle', 'rb'))

# print(list(input_df.to_numpy()[0]))
model_inputs=list(input_df.to_numpy()[0])

get_estimated_price(float(model_inputs[0]), model_inputs[1], model_inputs[2],model_inputs[3])

st.subheader('Prediction')
predicted_price=get_estimated_price(float(model_inputs[0]), model_inputs[1], model_inputs[2],model_inputs[3])

html_str = f"""
<h3 style="color:lightgreen;">{predicted_price} LAKH</h3>
"""

st.markdown(html_str, unsafe_allow_html=True)
#st.write(predicted_price,'LAKH')



st.info("""**Note: ** [The data source is]: ** (https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data). the following steps have been applied till we reached the model:

        1- Data Acquisition/ Data Collection (reading data, adding headers)

        2- Data Cleaning / Data Wrangling / Data Pre-processing (handling missing values, correcting data fromat/ data standardization 
        or transformation/ data normalization/ data binning/ Preparing Indicator or binary or dummy variables for Regression Analysis/ 
        Saving the dataframe as ".csv" after Data Cleaning & Wrangling)
        
        3- Exploratory Data Analysis (Analyzing Individual Feature Patterns using Visualizations/ Descriptive statistical Analysis/ 
        Basics of Grouping/ Correlation for continuous numerical variables/ Analysis of Variance-ANOVA for ctaegorical or nominal or 
        ordinal variables/ What are the important variables that will be used in the model?)
        
        4- Model Development (Single Linear Regression and Multiple Linear Regression Models/ Model Evaluation using Visualization)
        
        5- Polynomial Regression Using Pipelines (one-dimensional polynomial regession/ multi-dimensional or multivariate polynomial 
        regession/ Pipeline : Simplifying the code and the steps)
        
        6- Evaluating the model numerically: Measures for in-sample evaluation (Model 1: Simple Linear Regression/ 
        Model 2: Multiple Linear Regression/ Model 3: Polynomial Fit)
        
        7- Predicting and Decision Making (Prediction/ Decision Making: Determining a Good Model Fit)
        
        8- Model Evaluation and Refinement (Model Evaluation/ cross-validation score/ over-fitting, under-fitting and model selection)

""")


with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Published By: <a href="https://golytics.github.io/" target="_blank">Dr. Mohamed Gabr</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)



#streamlit_analytics.stop_tracking(unsafe_password="forward1")
