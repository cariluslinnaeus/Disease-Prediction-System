"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request
from Disease_Prediction_Website import app

@app.route('/home')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact Us',
        year=datetime.now().year,
        message='Mail us.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Welcome.'
    )

@app.route('/predict')
def predict():
    """Renders the about page."""
    return render_template(
        'predict.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
samples =[]
@app.route('/predictions',methods=['POST','GET'])
def predictions():
    if request.method == "POST":
        import pandas as pd
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        
        data = pd.read_csv('dataset.csv')
        symptoms = pd.read_csv('Symptom-severity.csv',header=0)
        precautions = pd.read_csv('symptom_precaution.csv')
        symptoms.shape
        
        description = pd.read_csv('symptom_Description.csv')
        
        #Removing trailing white spaces
        cols = data.columns
        df = data[cols].values.flatten()
        s = pd.Series(df)
        s = s.str.strip()
        s = s.values.reshape(data.shape)
        
        #creating a new clean dataset
        data = pd.DataFrame(s, columns=data.columns)
        data = data.fillna(0)
        
        #Extracting values of the dataframe
        vals = data.values
        sympts = symptoms['Symptom'].unique()
        symptsID = symptoms['ID']
        
        for i in range(len(sympts)):
            vals[vals==sympts[i]] = symptoms[symptoms['Symptom']==sympts[i]]['weight'].values[0]
        
        
        data1 = pd.DataFrame(vals,columns=cols)
        
        #Giving symptoms not included in sumptoms datasets IDs
        data1 = data1.replace('dischromic _patches',0)
        data1 = data1.replace('foul_smell_of urine',0)
        data1 = data1.replace('spotting_ urination',0)
        data1 = data1.replace('foul_smell_of urine',0)
        
        #Labels and Features
        features = data1.iloc[:,1:]
        labels =data1['Disease'].values
        
        #Test Train Split
        x_train,x_test,y_train,y_test=train_test_split(features,labels,train_size=0.85,shuffle =True)
        
        #Model SVC
        Model_SVC = SVC()
        Model_SVC.fit(x_train, y_train)
        
        #Test
        predicted =Model_SVC.predict(x_test)
        
        accuracy_score(y_test,predicted)
        
        #Model RFC
        RFC = RandomForestClassifier()
        RFC.fit(x_train,y_train)
        
        predt = RFC.predict(x_test)
        
        accuracy = accuracy_score(y_test,predt) #1.0
        #mpredict
        dat = np.array((10,22,3,4,5,0,30,11,0,0,0,40,0,12,34,0,0))
        
        predictip = Model_SVC.predict(dat.reshape(1,-1))
        pdtip  =RFC.predict(dat.reshape(1,-1))
        s1 = request.form.get('s1',type=int)
        s2 = request.form.get('s2',type=int)
        s3 = request.form.get('s3',type=int)
        s4 = request.form.get('s4',type=int)
        s5 = request.form.get('s5',type=int)
        s6 = request.form.get('s6',type=int)
        s7 = request.form.get('s7',type=int)
        s8 = request.form.get('s8',type=int)
        s9 = request.form.get('s9',type=int)
        s10 = request.form.get('s10',type=int)
        s11 = request.form.get('s11',type=int)
        s12 = request.form.get('s12',type=int)
        s13 = request.form.get('s13',type=int)
        s14 = request.form.get('s14',type=int)
        s15 = request.form.get('s15',type=int)
        s16 = request.form.get('s16',type=int)
        s17 = request.form.get('s17',type=int)
        samples = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17]
        samples = np.array(samples)
        
        #Preprocessing input: Relacing symptom Ids with symptom weights
        for i in range(len(symptsID)):
             samples[samples==symptsID[i]] = symptoms[symptoms['ID']==symptsID[i]]['weight'].values[0]
        #Predicting the disease
        disease = Model_SVC.predict(samples.reshape(1,-1))
        desc = description[description['Disease']==disease[0]]['Description'].values[0]
        prec1 = precautions[precautions['Disease']==disease[0]]['Precaution_1'].values[0]
        prec2 = precautions[precautions['Disease']==disease[0]]['Precaution_2'].values[0]
        prec3 = precautions[precautions['Disease']==disease[0]]['Precaution_3'].values[0]
        prec4 = precautions[precautions['Disease']==disease[0]]['Precaution_4'].values[0]
        status = all(i==0 for i in samples)
        
        if(status == False):
             return render_template('result.html',result=disease[0],accuracy=accuracy*100,desc=desc,prec1=prec1,prec2=prec2,prec3=prec3,prec4=prec4)
        
        else:
            return render_template('normal.html')
        


@app.route('/get_doctor')
def doctor():
    """Renders the doctors page."""
    return render_template(
        'doctors.html',
        title='Doctors',
        year=datetime.now().year,
        message='Welcome.'
    )


@app.route('/medical_history')
def history():
    """Renders the history page."""
    return render_template(
        'history.html',
        title='Medical History',
        year=datetime.now().year,
        message='Welcome.'
    )

@app.route('/')
def entry():
    """Renders the history page."""
    return render_template(
        'home.html',
        title='',
        year=datetime.now().year,
        message='Welcome.'
    )
@app.route('/model_info')
def model():
    """Renders the history page."""
    return render_template(
        'model.html',
        title='',
        year=datetime.now().year,
        message='Welcome.'
    )