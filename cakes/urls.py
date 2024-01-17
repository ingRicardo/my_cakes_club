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

   path('email/', views.send_welcome_email, name='send_welcome_email')

]