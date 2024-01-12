 
# Create your models here.

from django.db import models

class Cake(models.Model):
  caketype = models.CharField(max_length=255)
  cakeevent = models.CharField(max_length=255)
  size = models.IntegerField(null=True)
  shape = models.CharField(null =True)
  created_date =  models.DateField(null=True)
  
 # def __str__(self):
 #       return self.caketype
  
class CakeFinalJson(models.Model):
    jsondata = models.JSONField(null=True)
    created_date =  models.DateField(null=True)
    email = models.CharField(max_length=255, null=True)

class CakeComment(models.Model):
   name = models.CharField(max_length=255)
   comment = models.CharField(max_length=255)