from rest_framework import serializers
from .models import Cake, CakeFinalJson, CakeComment, CakesDataJson

# Create a model serializer
class CakesSerializer(serializers.HyperlinkedModelSerializer):
    # specify model and fields
    class Meta:
        model = Cake
        fields = ('caketype', 'cakeevent', 'size', 'shape', 'created_date')

class CakesJsonSerializer(serializers.HyperlinkedModelSerializer):
    # specify model and fields
    class Meta:
        model = CakeFinalJson
        fields = ('jsondata','created_date','email')

class CakesCommentSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = CakeComment
        fields = ('name', 'comment')

class CakesDataJsonSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = CakesDataJson
        fields =('jsoncakesdata', 'created_at')