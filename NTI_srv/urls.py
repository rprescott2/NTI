from django.contrib import admin
from django.conf.urls import url
from django.urls import path
from rest_framework.routers import DefaultRouter
from NTI_srv_neyronka import views


class NestedDefaultRouter(DefaultRouter):
    pass


urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^predict/$', views.Predicttion.as_view(), name='predict'),

]

router = DefaultRouter()
dict_router = DefaultRouter()
router.register(r'wind-turbine', views.WindTurbineViewSet, 'wind-turbine')
router.register(r'solar-panel', views.SolarPanelViewSet, 'solar-panel')
router.register(r'data', views.MeteoDataViewSet, 'data')
dict_router.register(r'building-type', views.BuildingTypeViewSet, 'building-type')
dict_router.register(r'building', views.BuildingViewSet, 'building')

urlpatterns += router.urls + dict_router.urls

