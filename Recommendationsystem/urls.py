"""Recommendationsystem URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import index
##from . import UserDashboard
##from . import index
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
     path('chat', index.chat),
     path('data', index.data),
     path('nutritient', index.nutritient),
     path('admindashboard', index.admindashboard),
     path('userdashboard', index.userdashboard),
     path('',index.dashboard),
     path('login', index.login),
     path('register', index.register),
     path('doregister', index.doregister),
     path('dologin', index.dologin),
     path('viewuser', index.viewuser),
     path('getcity', index.getcity),
     path('temp', index.temp),
     path('nutritient', index.nutritient),
     path('micronutritient', index.micronutritient),
     path('analyze', index.analyze),
               

    #path('index', index.inde),
    #path('aboutus',index.about),
    #path('service',index.service),
    #path('registration',index.doregister),
    #path('signup',index.dologin),
    
    #path('logout',index.logout),
    #path('viewuserprofile',index.viewuser),
    #path('viewpredicadmin',index.viewpredicadmin),
    #path('dashremove',index.dashremove),
    #path('index',index.index),
    #path('livepred',index.livepred),
    #path('prevpred',index.prevpred),
    #path('dashboard',index.dashboard),

    #path('myprofile',index.myprofile),
  
      
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

