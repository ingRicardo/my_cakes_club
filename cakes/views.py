 

from django.http import HttpResponse
from django.template import loader
from .models import Cake, CakeFinalJson, CakeComment
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework  import  viewsets
from  .serializers import CakesSerializer, CakesJsonSerializer, CakesCommentSerializer
from django.core.mail import EmailMultiAlternatives, send_mail
import json
from django.conf import settings

#def send_mail_func(request):
#  myemail= send_mail("New Order!", "Hey buddy, you have a new order",
#  "sender-email@gmail.com", ["reciever-email@gmail.com"])
#  template = loader.get_template('email.html')
#  context = {
#    'myemail' : myemail,
#  }
#  return HttpResponse(template.render(context, request))


#def send_mail_func(request):
#    send_mail("New Order!", "Hey buddy, you have a new order",
#          "sender-email@gmail.com", ["reciever-email@gmail.com"])
#    return HttpResponse("Email Sent")


def send_welcome_email(jsondata, email):
    subject = 'Welcome to My Cakes Site'
    workorder =""
    for key, value in jsondata.items():
      print(key, value)
      if (key == "uniqueId"):
          workorder = value
    message = "Your current data is : " +json.dumps(jsondata)  + "\n\n" +  "Your workOrder is: "+workorder  +"\n\n For details call or whatsapp to Riky 6641268391"
    from_email = 'ramo2884@gmail.com'
    recipient_list = [json.dumps(email),"ramo2884@gmail.com"]
    send_mail(subject, message, from_email, recipient_list)
    #return HttpResponse("Email Sent", request)
  #  return HttpResponse("Email Sent", request)



# create a viewset
class EmailViewSet(viewsets.ModelViewSet):
   varia ='email was sent'

def cakes(request):
  mycakes= Cake.objects.all().values()
  template = loader.get_template('all_cakes.html')
  context = {
    'mycakes' : mycakes,
  }
  return HttpResponse(template.render(context, request))

# create a viewset
class CakesViewSet(viewsets.ModelViewSet):
    # define queryset
    queryset = Cake.objects.all()
 
    # specify serializer to be used
    serializer_class = CakesSerializer

def cakesJson(request):
  mycakesjson= CakeFinalJson.objects.all().values()
  template = loader.get_template('all_cakesjson.html')
  context = {
    'mycakesjson' : mycakesjson,
  }
  return HttpResponse(template.render(context, request))

class CakesJsonViewSet(viewsets.ModelViewSet):
    # define queryset
    queryset = CakeFinalJson.objects.all()
 
    # specify serializer to be used
    serializer_class = CakesJsonSerializer

@api_view(['POST'])
def postJsonCake(request):
    serializer = CakesJsonSerializer(data=request.data)
 
    if serializer.is_valid():
        serializer.save()
        json  = serializer.data['jsondata']
        email  = serializer.data['email']
        send_welcome_email(json, email)
        print(json)
        print(email)
    return Response(serializer.data)

def cakesComment(request):
  cakescomment = CakeComment.objects.all().values()
  template = loader.get_template('all_cakescomments.html')
  context = {
    'cakescomment' : cakescomment,
    }
  return HttpResponse(template.render(context, request))

class CakesCommentViewSet(viewsets.ModelViewSet):
    # define queryset
    queryset = CakeComment.objects.all()
 
    # specify serializer to be used
    serializer_class = CakesCommentSerializer

@api_view(['POST'])
def postCommentCake(request):
    serializer = CakesCommentSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
    return Response(serializer.data)