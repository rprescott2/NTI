from django.http import HttpResponse
from rest_framework import status
from rest_framework import viewsets
from rest_framework.views import APIView
from NTI_srv_neyronka import models, utils
import math

from .serializers import *
from NTI_srv_neyronka import models as neyron_model
import numpy as np
import pandas as pd


class MeteoDataViewSet(viewsets.ModelViewSet):

    def get_serializer_class(self):
        if hasattr(self, 'action'):
            if self.action == 'list':
                return MeteoDataListSerializer
        return MeteoDataDetailSerializer

    def get_queryset(self):
        qs = neyron_model.MeteoData.objects.all()
        return qs


class ActualDataViewSet(viewsets.ModelViewSet):

    def get_serializer_class(self):
        if hasattr(self, 'action'):
            if self.action == 'list':
                return ActualDataListSerializer
        return ActualDataDetailSerializer

    def get_queryset(self):
        qs = neyron_model.ActualData.objects.all()
        return qs


class WindTurbineViewSet(viewsets.ModelViewSet):

    def get_serializer_class(self):
        if hasattr(self, 'action'):
            if self.action == 'list':
                return WindTurbineListSerializer
        return WindTurbineDetailSerializer

    def get_queryset(self):
        qs = neyron_model.WindTurbine.objects.all()
        return qs


class SolarPanelViewSet(viewsets.ModelViewSet):

    def get_serializer_class(self):
        if hasattr(self, 'action'):
            if self.action == 'list':
                return SolarPanelListSerializer
        return SolarPanelDetailSerializer

    def get_queryset(self):
        qs = neyron_model.SolarPanel.objects.all()
        return qs


class BuildingTypeViewSet(viewsets.ModelViewSet):
    serializer_class = BuildingTypeListSerializer

    def get_queryset(self):
        qs = neyron_model.BuildingType.objects.all()
        return qs


class BuildingViewSet(viewsets.ModelViewSet):

    def get_serializer_class(self):
        if hasattr(self, 'action'):
            if self.action == 'list':
                return BuildingListSerializer
        return BuildingDetailSerializer

    def get_queryset(self):
        qs = neyron_model.Building.objects.all()
        return qs


class Predicttion(APIView):
    def get(self, request, *args, **kwargs):
        if self.request.query_params.get('df'):
            pass
        else:
            data_for_predict = np.array([self.request.query_params.get('prectot'),
                                     self.request.query_params.get('qv2m'),
                                     self.request.query_params.get('ps'),
                                     self.request.query_params.get('t2m')]
                                    ).astype(float).reshape(1, -1)
        Feature_scaler, Target_scaler = utils.scales()
        packed_allsky = utils.pack_allsky(data_for_predict, Feature_scaler)
        allsky = utils.allsky_predict(packed_allsky, Target_scaler)
        pack = utils.pack_wind(data_for_predict)
        pack = np.append(pack, allsky, axis=1)
        wind = utils.wind_predict(pack)
        turbine = models.WindTurbine.objects.filter(model=self.request.query_params.get('turbine_model')).first()
        solar = models.SolarPanel.objects.filter(model=self.request.query_params.get('solar_model')).first()
        turbine_Kw = 0
        solar_KW = 0
        if len(allsky) == 1:
            turbine_Kw = 0.5*math.pi*float(turbine.Q)*(float(turbine.diameter)**2)*(wind**3)*float(turbine.efficiency)*float(turbine.Ng)/1000 if turbine else 0
            solar_KW = allsky*float(solar.Ko)*float(solar.power)*float(solar.efficiency)/float(solar.U) if solar else 0
        else:
            for i in range(len(allsky)):
                turbine_Kw += 0.5 * math.pi * float(turbine.Q) * (float(turbine.diameter) ** 2) * (wind[i] ** 3) * float(
                    turbine.efficiency) * float(turbine.Ng) / 1000
                solar_KW += allsky[i] * float(solar.Ko) * float(solar.power) * float(solar.efficiency) / float(solar.U)
        return HttpResponse('solar_KW:{},wind_KW:{},allsky:{},wind:{}'.format(solar_KW,turbine_Kw, allsky, wind), status=status.HTTP_200_OK)
