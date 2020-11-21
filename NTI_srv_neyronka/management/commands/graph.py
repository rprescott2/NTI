import math

from django.core.management.base import BaseCommand
from sklearn.model_selection import train_test_split
from sklearn.svm._libsvm import cross_validation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.utils import np_utils
from tensorflow.examples.saved_model.integration_tests.mnist_util import INPUT_SHAPE
from statsmodels.tsa import stattools
from tensorflow.python.keras.models import Model
from itertools import zip_longest

from NTI_srv_neyronka import models as neyron_models
import pandas as pd
import numpy as np

import datetime
from itertools import zip_longest
import matplotlib.pyplot as plt
from NTI_srv_neyronka.models import  *


class Command(BaseCommand):
    def handle(self, *args, **options):
        turbine1 = WindTurbine.objects.get(model='Condor Air 20')
        turbine2 = WindTurbine.objects.get(model='Condor Air 30')
        turbine3 = WindTurbine.objects.get(model='Condor Air 50')
        solar1 = SolarPanel.objects.get(model='Hevel HVL')
        solar2 = SolarPanel.objects.get(model='Top Ray TPS-MGV(72DH)')
        solar3 = SolarPanel.objects.get(model='Delta SM 260-24P')

        # Condor Air 50
        speed = [0.1, 0.5,1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
        t1 = []
        t2 = []
        t3 = []
        s1 = []
        s2 = []
        s3 = []
        for i in speed:
            t1.append(0.5 * math.pi * float(turbine1.Q) * (float(turbine1.diameter) ** 2) * (i ** 3) * float(
                    turbine1.efficiency) * float(turbine1.Ng) / 1000)
            t2.append(0.5 * math.pi * float(turbine2.Q) * (float(turbine2.diameter) ** 2) * (i ** 3) * float(
                turbine2.efficiency) * float(turbine2.Ng) / 1000)
            t3.append(0.5 * math.pi * float(turbine3.Q) * (float(turbine3.diameter) ** 2) * (i ** 3) * float(
                turbine3.efficiency) * float(turbine3.Ng) / 1000)
        for i in speed:
            s1.append(i * float(solar1.Ko) * float(solar1.power) * float(solar1.efficiency) / float(
                solar1.U))
            s2.append(i * float(solar2.Ko) * float(solar2.power) * float(solar2.efficiency) / float(
                solar2.U))
            s3.append(i * float(solar3.Ko) * float(solar3.power) * float(solar3.efficiency) / float(
                solar3.U))
        fig, ax = plt.subplots()

        ax.plot(speed, s1)
        ax.plot(speed, s2)
        ax.plot(speed, s3)
        ax.legend(['Hevel HVL','Top Ray TPS-MGV(72DH)','Delta SM 260-24P'])

        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title('Вырабатываемая мощность панели')
        ax.set_xlabel('Инсоляция')
        ax.set_ylabel('Вырабатываемая мощность (КВт)')

        plt.show()