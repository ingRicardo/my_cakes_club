from django.urls import include, path

from rest_framework  import routers

from . import views

from .views import *

router = routers.DefaultRouter()

# define the router path and viewset to be used
#router.register(r'email', send_welcome_email)
router.register(r'cakes', CakesViewSet)
router.register(r'cakesjson', CakesJsonViewSet)
router.register(r'cakescomments', CakesCommentViewSet)

router.register(r'cakesdatajson', CakesDataJsonViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('cakes/', views.cakes, name='cakes'),
    path('cakesjson/', views.cakesJson, name='cakesjson'),
    path('cakesdatajson/', views.cakesdataJson, name='cakesdatajson'),
    path('api-cakes/', include('rest_framework.urls')),
    path('postjsoncake/',views.postJsonCake),
    path('cakescomments/',views.cakesComment, name='cakescomments'),
    path('postcommentcake/',views.postCommentCake),

   path('email/', views.send_welcome_email, name='send_welcome_email'),
   
   path('neuralnets/', views.callNeuralNets),
   path('neuralnets/lifmodel/', views.LifModelFunc , name="lifmodel"),

   path('lifsnn/', views.showDataset),
   path('lifsnn/receptivefields/', views.receptiveFields),
   path('lifsnn/latancy/', views.latancy),
   path('lifsnn/latancy2/', views.latancy2),
   path('lifsnn/presynapneurons/', views.presynapneurons),
   path('lifsnn/postsynapneurons/', views.postsynaptic),
   path('lifsnn/synapticspikes/', views.synapticspikes),

   path('lifsnn/dataset', views.getIrisDataset, name='dataset'),
   path('lifsnn/gaussian', views.getReceptiveFields, name='gaussian'),

   path('lifsnn/irisdataimg', views.getIrisDatasetImage, name='irisdatimg'),
   path('lifsnn/irisgaussian', views.receptiveFieldsJson, name='irisgaussian'),
   path('lifsnn/irislatancy', views.latancyJson, name='irislatancy'),
   path('lifsnn/irislatancy2', views.latancyJson2, name='irislatancy2'),
   path('lifsnn/irispresynapneurons', views.presynapneuronsJson, name='irispresynapneurons'),
   path('lifsnn/irispostsynneu', views.postsynapticJson, name='irispostsynapneurons'),
   path('lifsnn/irissynapticspikes', views.synapticspikesJson, name='irissynapticspikes'),
]