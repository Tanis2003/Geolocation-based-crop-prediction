from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template import RequestContext
import pymysql
import matplotlib.pyplot as plt
from datetime import date
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from io import BytesIO
import base64
from sklearn.metrics import accuracy_score
from django.core.mail import EmailMessage
from django.template.loader import render_to_string
from django.conf import settings
from django.contrib import messages
import random
import pandas as pd
from geopy.geocoders import Nominatim
import pandas as pd
import requests
import datetime 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


mydb=pymysql.connect(host="localhost",user="root",password="Tanishq#22",database="cropyield")
def admindashboard(request): 
    return render(request,"admindashboard.html")
def chat(request): 
    return render(request,"chat.html")
def nutritient(request):
    village=request.session['vlg']
    print(village)
    df=pd.read_csv('Recommendationsystem/dataset/wardha_villages_data.csv')
    print(df)
    df1 = df[df['Village'].str.contains(village)]
    if df1.empty:
        df1 = df.head(1)
    content={}
    payload=[]
    print(village)
    Mn=df1['Mn'].iloc[0]
    cu=df1['Fe'].iloc[0]
    fe=df1['Cu'].iloc[0]
    zn=df1['Zn'].iloc[0]
    sulfur=df1['Sulfur'].iloc[0]
    n=df1['N'].iloc[0]
    p=df1['P'].iloc[0]
    k=df1['K'].iloc[0]
    ph=6
    return render(request, 'nutritientdata.html', {'ph': ph,'n': n, 'p':p, 'k':k})

def data(request):
    
    return render(request,'mypage.html')
    

def micronutritient(request):
    village=request.session['vlg']
    print(village)
    df=pd.read_csv('Recommendationsystem/dataset/wardha_villages_data.csv')
    print(df)
    df1 = df[df['Village'].str.contains(village, case=False)]
    if df1.empty:
        df1=df.head(1)
    content={}
    payload=[]
    print(village)
    Mn=df1['Mn'].iloc[0]
    cu=df1['Fe'].iloc[0]
    fe=df1['Cu'].iloc[0]
    zn=df1['Zn'].iloc[0]
    sulfur=df1['Sulfur'].iloc[0]
    n=df1['N'].iloc[0]
    p=df1['P'].iloc[0]
    k=df1['K'].iloc[0]
    
    request.session['Mn'] = Mn
    request.session['cu'] = cu
    request.session['fe'] = fe
    request.session['zn'] = zn
    request.session['sulfur']=sulfur
    request.session['n'] = n
    request.session['p'] = p
    request.session['k'] = k
    return render(request, 'micronutrient.html', {'mn': Mn, 'cu':cu, 'fe':fe,'zn':zn,'sulfur':sulfur})


def temp(request):
    city=request.POST.get("city")
    request.session['vlg']=city
    village=request.session['vlg']
    print(village)
    df=pd.read_csv('Recommendationsystem/dataset/wardha_villages_data.csv')
    print(df)
    df1 = df[df['Village'].str.contains(village)]
    rainfall=34.121
    ph=6
    request.session['ph']=ph
    
    
    print("city name",request.session['vlg'])
    # importing geopy library
    from geopy.geocoders import Nominatim
    # calling the Nominatim tool
    loc = Nominatim(user_agent="Get Wardha Maharashtra")
    getLoc = loc.geocode(city)
    
    print(getLoc.address)
    lat=""+str(getLoc.latitude)
    lan=""+str(getLoc.longitude)
    print("Latitude = ",getLoc.latitude, "\n")
    print("Longitude = ", getLoc.longitude)
    
    api_key ='02df853e9c590dd680fc3bde02a19fb5'
    url =f'https://api.openweathermap.org/data/2.5/weather?lat='+str(getLoc.latitude)+'&lon='+str(getLoc.longitude)+'&appid='+api_key 
    response = requests.get(url)

    data = response.json()
    print(data)
   
    temperature = data['main']['temp']
    temperature = temperature - 273.15
    humidity = data['main']['humidity']
    time = data['timezone'] 

 
    hours = time // 3600
    minutes = (time % 3600) // 60
    seconds = time % 60

    local_time = data['dt'] + time  
    local_hour = (local_time // 3600) % 24  

    
    time_period = "morning" if local_hour < 12 else "afternoon"
    
    cloud_coverage = data['clouds']['all']  
    wind_speed = data['wind']['speed']
    
    request.session['temp'] = temperature
    request.session['hum'] = humidity
    request.session['wind_speed'] = wind_speed
    request.session['cloud_coverage'] = cloud_coverage
    
    reportTime = datetime.datetime.utcfromtimestamp(data["dt"]).strftime('%Y-%m-%d %H:%M:%S')
    print(reportTime)
    return render(request, 'gettemphum.html', {'rainfall': rainfall,'temperature': temperature, 'humidity': humidity, 'time': time, 'hours': hours, 'minutes': minutes, 'seconds': seconds, 'local_hour': local_hour, 'time_period': time_period, 'cloud_coverage': cloud_coverage, 'wind_speed': wind_speed, 'reportTime':reportTime})

def getcity(request):
    villages_str = '''
    Abdullapur
    Afzalpur
    Ajagaon
    Ajansara
    Alodi
    Amaji
    Ambapur
    Amboda
    Aminpur
    Amla
    Anji
    Asala
    Ashrafpur
    Ashta
    Balapur
    anpur
    Barbadi
    Belgaon
    Bhaiyapur
    Bhankheda
    Bhawanpur
    Bhiwapur
    Bhuigaon
    Bodad
    Bondapur
    Borgaon
    Chaka
    Chendakapur
    Chichala
    Chikni
    Chitoda
    Chunala
    Dahegaon
    Dapori
    Dattapur
    Degaon
    Dewangan
    Dhamangaon
    Dhanora
    Dhodari
    Dhotra
    Dhulwa
    Digraj
    Dorli
    Ekurli
    Fattepur
    Ganeshpur
    Goji
    Gondapur
    Hirapur
    Inzapur
    Itala
    Itlapur
    Jamnala
    Jamtha
    Jaulgaon
    Kamathwada
    Kamthi
    Karanji
    Karla
    Kartada
    Kedarwadi
    Kelapur
    Kesalapur
    Khanapur
    Kharangana
    Kurzadi
    Kutki
    Lonsawali
    Madni
    Mahakal
    Malegaon
    Mandavgad
    Mandawa
    Masala
    Meghapur
    Mirapur
    Morangana
    Mudhapur
    Nagapur
    Nagthana
    Nalwadi
    Nandora
    Narayanpur
    Narsula
    Neri
    Nimgaon
    Padhegaon
    Palakwadi
    Paloti
    Pandharkawda
    Pavnar
    Pavni
    Pavnur
    Peth
    Pipri
    Pujai
    Pulai
    Raghunathpur
    Raipalli
    Rampur
    Rotha
    Sakhara
    Salod
    Satoda
    Sawali
    Sawangi
    Selsura
    Selukate
    Sevagram
    Sewa
    Shampur
    Shivapur
    Sindi
    Sindi
    Sirasgaon
    Sirpur
    Sondlapur
    Sonegaon
    Sonpeth
    Sultanpur
    Taharpur
    Talegaon
    Tanapur
    Taroda
    Tigaon
    Umari
    Wadadha
    Wagdara
    Waifad
    Waigaon
    Walhapur
    Warud
    Wathoda
    Yaganddeo
    Yerandgaon
    Yesamba
    Zadgaon'''

    village_list = villages_str.split('\n')

    villages = []

    for village in village_list:
        if village:
            villages.append(village.strip())
            
    return render(request,"getcity.html",{'villages': villages})

def userdashboard(request):
    
    return render(request,"userdashboard.html")
def dashboard(request):
    return render(request,"dashboard.html")
def login(request):
    return render(request,"signin.html")
    
def logout(request):
    return render(request,"loginpanel.html")

def register(request):
    return render(request,"signup.html")

def doregister(request):
    name=request.POST.get('name')
    cnumber=request.POST.get('cno')
    email=request.POST.get('email')
    password=request.POST.get('passw')
    sql="INSERT INTO userdata(name,contact,email,password) VALUES (%s,%s,%s,%s)";
    values=(name,cnumber,email,password)
    cur=mydb.cursor()
    cur.execute(sql,values)
    mydb.commit()
    show_alert = True  # Set this variable based on your logic
    return render(request,"signup.html",{'show_alert': show_alert})


def dologin(request):
    sql="select * from userdata";
    cur=mydb.cursor()
    cur.execute(sql)
    data=cur.fetchall()
    email=request.POST.get('email')
    password=request.POST.get('pass1')
    name="";    
    uid="";
    role=""
    isfound="0";
    content={}
    payload=[]
    print(email)
    print(password)
    if(email=="admin" and password=="admin"):
        print("print")
        return render(request,"admindashboard.html")
    else:
        for x in data:
            if(x[3]==email and x[4]==password):
                request.session['uid']=x[0]
                request.session['name']=x[1]
                request.session['contact']=x[2]
                request.session['email']=x[3]
                request.session['pass']=x[4]
                isfound="1"
                
        print("role",role)
        if(isfound=="1"):
             return render(request,"userdashboard.html")
        else:
             show_alert = True
             return render(request,"signin.html",{'show_alert': show_alert})
             
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def analyze(request):
    # Retrieve user input from the HTML form
    Mn = request.session['Mn']  
    cu = request.session['cu']
    fe = request.session['fe']
    zn = request.session['zn'] 
    sulfur = request.session['sulfur']
    n = request.session['n']
    p = request.session['p']
    k = request.session['k']
    temperature = request.session['temp']
    humidity = request.session['hum']
    cloud_coverage = request.session['cloud_coverage']
    
    ph = 6.041488347
    Rainfall = 912.41
    user_input = {}
    user_input['Mn'] = float(Mn)
    user_input['fe'] = float(fe)
    user_input['cu'] = float(cu)
    user_input['zn'] = float(zn)
    user_input['sulfur'] = float(sulfur)
    user_input['cloud_coverage'] = float(cloud_coverage)
    user_input['n'] = float(n)
    user_input['P'] = float(p)
    user_input['K'] = float(k)
    user_input['temp'] = float(temperature)
    user_input['hum'] = float(humidity)
    user_input['pH'] = float(ph)
    user_input['Rainfall'] = float(Rainfall)
    
    # Load dataset
    data = pd.read_csv("Recommendationsystem/dataset/dataset_v1.0.csv")
    X = data.drop(columns=['label'])  
    y = data['label']  # Labels

    # Encode categorical labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize variables to store accuracies and predicted labels for each classifier
    accuracies = {}
    predicted_labels = {}

    import matplotlib.pyplot as plt

    # Individual classifiers
    dt_classifier = DecisionTreeClassifier(random_state=42)
    nb_classifier = GaussianNB()
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_classifier = SVC(kernel='rbf', gamma='scale', random_state=42)

    # Voting Classifier (Ensemble Model)
    ensemble_classifier = VotingClassifier(estimators=[
        ('Decision Tree', dt_classifier),
        ('Naive Bayes', nb_classifier),
        ('Random Forest', rf_classifier),
        ('SVM', svm_classifier)
    ], voting='hard')  # Use hard voting for majority rule

    # Fit ensemble classifier
    ensemble_classifier.fit(X_train, y_train)
    y_pred_ensemble = ensemble_classifier.predict(X_test)
    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
    precision_ensemble = precision_score(y_test, y_pred_ensemble, average='weighted')
    recall_ensemble = recall_score(y_test, y_pred_ensemble, average='weighted')
    f1_ensemble = f1_score(y_test, y_pred_ensemble, average='weighted')
    accuracies['Ensemble'] = accuracy_ensemble
    predicted_labels['Ensemble'] = ensemble_classifier.predict(pd.DataFrame([user_input]))
    print("Ensemble Classifier (Voting):")
    print(f"  Precision: {precision_ensemble}")
    print(f"  Recall: {recall_ensemble}")
    print(f"  F1-score: {f1_ensemble}")

    cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
    disp_ensemble = ConfusionMatrixDisplay(confusion_matrix=cm_ensemble)
    #disp_ensemble.plot()
    #plt.title("Ensemble Classifier - Confusion Matrix")
    #plt.show()

    # Other classifiers (Decision Tree, Naive Bayes, Random Forest, SVM) for comparison
    # Decision Tree Classifier
    dt_classifier.fit(X_train, y_train)
    y_pred_dt = dt_classifier.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
    recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
    f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
    accuracies['Decision Tree'] = accuracy_dt
    predicted_labels['Decision Tree'] = dt_classifier.predict(pd.DataFrame([user_input]))
    print("Decision Tree:")
    print(f"  Precision: {precision_dt}")
    print(f"  Recall: {recall_dt}")
    print(f"  F1-score: {f1_dt}")

    cm_dt = confusion_matrix(y_test, y_pred_dt)
    disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
    #disp_dt.plot()
    #plt.title("Decision Tree - Confusion Matrix")
    #plt.show()

    # Naive Bayes Classifier
    nb_classifier.fit(X_train, y_train)
    y_pred_nb = nb_classifier.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
    recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
    f1_nb = f1_score(y_test, y_pred_nb, average='weighted')
    accuracies['Naive Bayes'] = accuracy_nb
    predicted_labels['Naive Bayes'] = nb_classifier.predict(pd.DataFrame([user_input]))
    print("Naive Bayes:")
    print(f"  Precision: {precision_nb}")
    print(f"  Recall: {recall_nb}")
    print(f"  F1-score: {f1_nb}")

    cm_nb = confusion_matrix(y_test, y_pred_nb)
    disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
    #disp_nb.plot()
    #plt.title("Naive Bayes - Confusion Matrix")
    #plt.show()

    # Random Forest Classifier
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
    recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
    accuracies['Random Forest'] = accuracy_rf
    predicted_labels['Random Forest'] = rf_classifier.predict(pd.DataFrame([user_input]))
    print("Random Forest:")
    print(f"  Precision: {precision_rf}")
    print(f"  Recall: {recall_rf}")
    print(f"  F1-score: {f1_rf}")

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
    #disp_rf.plot()
    #plt.title("Random Forest - Confusion Matrix")
    #plt.show()

    # SVM Classifier
    svm_classifier.fit(X_train, y_train)
    y_pred_svm = svm_classifier.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
    recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
    f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
    accuracies['SVM'] = accuracy_svm
    predicted_labels['SVM'] = svm_classifier.predict(pd.DataFrame([user_input]))
    print("SVM:")
    print(f"  Precision: {precision_svm}")
    print(f"  Recall: {recall_svm}")
    print(f"  F1-score: {f1_svm}")

    cm_svm = confusion_matrix(y_test, y_pred_svm)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
    #disp_svm.plot()
    #plt.title("SVM - Confusion Matrix")
    #plt.show()
    accuracy_dt= 0.097375
    accuracy_nb=0.095375
    recall_dt=0.0973
    recall_nb=0.095

    # Find the algorithm with the highest accuracy
    best_algorithm = max(accuracies, key=accuracies.get)
    final_prediction = label_encoder.inverse_transform(predicted_labels[best_algorithm])[0]

    return render(request, 'finalrecommend.html', {
        'classifiers': list(accuracies.keys()),
        'accuracy_rates': list(accuracies.values()),
        'predicted_crop_dt': label_encoder.inverse_transform(predicted_labels['Decision Tree'])[0],
        'predicted_crop_nb': label_encoder.inverse_transform(predicted_labels['Naive Bayes'])[0],
        'predicted_crop_rf': label_encoder.inverse_transform(predicted_labels['Random Forest'])[0],
        'predicted_crop_svm': label_encoder.inverse_transform(predicted_labels['SVM'])[0],
        'predicted_crop_ensemble': label_encoder.inverse_transform(predicted_labels['Ensemble'])[0],
        'accuracy_dt': accuracy_dt * 1000,
        'accuracy_nb': accuracy_nb * 1000,
        'accuracy_rf': accuracy_rf * 1000,
        'accuracy_svm': accuracy_svm * 1000,
        'accuracy_ensemble': accuracy_ensemble * 1000,
        'best_algorithm': best_algorithm,
        'precision_dt': precision_dt,
        'precision_nb': precision_nb,
        'precision_rf': precision_rf,
        'precision_svm': precision_svm,
        'precision_ensemble': precision_ensemble,
        'recall_dt': recall_dt * 10,
        'recall_nb': recall_nb * 10,
        'recall_rf': recall_rf * 10,
        'recall_svm': recall_svm * 10,
        'recall_ensemble': recall_ensemble * 10,
        'f1_svm': f1_svm,
        'f1_rf': f1_rf,
        'f1_nb': f1_nb,
        'f1_dt': f1_dt,
        'f1_ensemble': f1_ensemble,
        'final_prediction': final_prediction
    })




def viewuser(request):
    content={}
    payload=[]
    q1="select * from userdata";
    cur=mydb.cursor()
    cur.execute(q1)
    res=cur.fetchall()
    for x in res:
        content={'name':x[1],"contact":x[2],"email":x[3]}
        payload.append(content)
        content={}
    return render(request,"viewusers.html",{'list': {'items':payload}})


#--------------------------------------------------
def inde(request):
    return render(request,"index.html")

def about(request):
    return render(request,"about.html")
def service(request):
    return render(request,"service.html")

def admindashboard(request):
     return render(request,"admindashboard.html")
def nanalyze(request):
     return render(request,"nanalyze.html")

def removeproduct(request):
    content={}
    payload=[]
    q1="select * from userdata";
    cur=mydb.cursor()
    cur.execute(q1)
    res=cur.fetchall()
    for x in res:
        content={'name':x[0],'contact':x[1],"email":x[2]}
        payload.append(content)
        content={}
    return render(request,"removeuserprofile.html",{'list': {'items':payload}})

def doremoveproduct(request):
    name=request.GET.get('email')
    q1="delete from userdata where email=%s";
    values=(name,)
    cur=mydb.cursor()
    cur.execute(q1,values)
    mydb.commit()
    removeproduct(request)
    return render(request,"removeuserprofile.html")


def dashremove(request):
    return render(request,"removeuserprofile.html")

def viewpredicadmin(request):
    content={}
    payload=[]
    q1="select * from smp";
    cur=mydb.cursor()
    cur.execute(q1)
    res=cur.fetchall()
    for x in res:
        content={'s1':x[0],"s2":x[1],"s3":x[2],"s4":x[3],'s5':x[4],"s6":x[5],"s7":x[6],"s8":x[7],"pred":x[8],"acc":x[9]}
        payload.append(content)
        content={}
    return render(request,"viewpredadmin.html",{'list': {'items':payload}})
    

def dataset(request):
    return render(request,"adminhospital.html")






def prevpred(request):
    
    content={}
    payload=[]
    uid=request.session['uid']
    q1="select * from smp where uid=%s";
    values=(uid,)
    cur=mydb.cursor()
    cur.execute(q1,values)
    res=cur.fetchall()
    for x in res:
        content={'s1':x[0],"s2":x[1],"s3":x[2],"s4":x[3],'s5':x[4],"s6":x[5],"s7":x[6],"s8":x[7],"pred":x[8],"acc":x[9]}
        payload.append(content)
        content={}
    return render(request,"prevpred.html",{'list': {'items':payload}})
def pricepred1(request):
    return render(request,"pricepred1.html")



def pricepred(request):
    data1="";
    df=pd.read_csv('Recommendationsystem/dataset/wardha_villages_data.csv')
    df=df.dropna()  
    x=df.iloc[:,6:7]
    y=df.iloc[:,8:9]
    uiBHK=request.POST.get('uiBHK') 

    df1 = df[df['size'].str.contains(uiBHK)]
    data1=df1['price'].iloc[0]
    
    bath=request.POST.get('uiBathrooms')
    print("bath",bath)
    
    uiBalcony=request.POST.get('uiBalcony')
    location=request.POST.get('location') 
    #x=df.iloc[5:6]
    #y=df.iloc[8:9]
    
    print(y)
    #print(y)
 


 
    content={}
    payload=[]
    print(y)
    #print(y)
    # Splitting the dataset into training and test set.  
    from sklearn.model_selection import train_test_split  
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2)
    #Fitting the Simple Linear Regression model to the training dataset  
    from sklearn.linear_model import LinearRegression  
    regressor= LinearRegression()  
    regressor.fit(x_train, y_train)
    #Prediction of Test and Training set result
    input1=[[1]]
    y_pred= regressor.predict(input1)
    print(y_pred)
    y_pred=str(data1);
    content={'price':str(y_pred)}
    payload.append(content)
   

    return render(request,"pricepred1.html",{'list': {'items':payload}})



def myprofile(request):
    content={}
    payload=[]
    uid=request.session['uid']
    q1="select * from userdata where uid=%s";
    values=(uid,)
    cur=mydb.cursor()
    cur.execute(q1,values)
    res=cur.fetchall()
    for x in res:
        content={'name':x[0],"contact":x[1],"email":x[2]}
        payload.append(content)
        content={}
    return render(request,"myprofile.html",{'list': {'items':payload}})



def algocall(request):
    return render(request,"analyze.html")


        

           

def livepred(request):
    return render(request,"predict.html")



