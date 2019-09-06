import pandas as pd
import numpy as np

def clean_New_Price(val):
    if type(val) is str:
        if ' Cr' in val:
            return float(val.replace(' Cr','')) * 100
        else:
            return float(val.replace(' Lakh',''))
    else:
        return 0

def clean_Kilometers_Driven(val):
    if val > 1000000:
        return val / 100
    else:
        return val

def map_location_type(val):
    dict_loc_map = {'Mumbai':'North',
                     'Pune':'North',
                     'Chennai':'South', 
                     'Coimbatore':'South',
                     'Hyderabad':'South',
                     'Jaipur':'North',
                     'Kochi':'South',
                     'Kolkata':'West',
                     'Delhi':'North',
                     'Bangalore':'South',
                     'Ahmedabad':'North',}
    return dict_loc_map[val]

def clean_data(data):
    data['Mileage'] = data['Mileage'].str.replace(' kmpl','')
    data['Mileage'] = pd.to_numeric(data['Mileage'].str.replace(' km/kg',''))

    data['Engine'] = pd.to_numeric(data['Engine'].str.replace(' CC',''))
    data['Power'] = pd.to_numeric(data['Power'].str.replace(' bhp',''), errors='coerce')

    data['Seats'] = data['Seats'].replace(0, 4)

    data['New_Price'] = data['New_Price'].apply(clean_New_Price)
    data['Location_Type'] = data['Location'].apply(map_location_type)
    data['Kilometers_Driven'] = data['Kilometers_Driven'].apply(clean_Kilometers_Driven)

    data['Brand'] = data['Name'].str.split(' ').str[0].str.lower().tolist()
    data['Model'] = data['Name'].str.split(' ').str[1:].str.join(' ').str.strip().str.lower()
    
    return data

def score(y_pred, y_true):
   error = np.square(np.log10(y_pred +1) - np.log10(y_true +1)).mean() ** 0.5
   score = 1 - error
   return score
