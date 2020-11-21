from django.contrib import admin
from NTI_srv_neyronka import models as neyronka_models

admin.site.register(neyronka_models.SolarPanel)
admin.site.register(neyronka_models.WindTurbine)
admin.site.register(neyronka_models.BuildingType)
admin.site.register(neyronka_models.Building)
admin.site.register(neyronka_models.Settings)
admin.site.register(neyronka_models.MeteoData)
admin.site.register(neyronka_models.ActualData)

# Register your models here.
