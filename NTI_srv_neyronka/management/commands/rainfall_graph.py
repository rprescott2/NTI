from itertools import zip_longest

import matplotlib.pyplot as plt
import datetime
from django.core.management.base import BaseCommand
import pandas as pd
import matplotlib.pyplot as plt


class Command(BaseCommand):
    def handle(self, *args, **options):
        dataframe = pd.read_csv('NTI_srv_neyronka/management/commands/POWER_SP.csv')
        years = dataframe['YEAR']
        months = dataframe.pop('MO')
        days = dataframe.pop('DY')
        dates = []
        data_plt = []
        for year, month, day in zip_longest(years, months, days):
            dates.append('%s-%s-%s 00:00' % (year, month, day))
            data_plt.append(datetime.datetime(year, month, day))
        dates = pd.DataFrame({'date': dates})

        dataframe['YEAR'] = dates
        dataframe.rename(columns={'YEAR': 'DATE'}, inplace=True)
        ALLSKY_SFC_SW_DWN = dataframe['ALLSKY_SFC_SW_DWN']
        remove_row = []
        for i in range(len(ALLSKY_SFC_SW_DWN)):
            if ALLSKY_SFC_SW_DWN[i] < 0:
                remove_row.append(i)

        T2M = dataframe['T2M']
        for i in range(len(T2M)):
            if T2M[i] < -6 and i not in remove_row:
                remove_row.append(i)

        dataframe.drop(index=remove_row, inplace=True)

        plt.plot(dataframe['DATE'], dataframe['PRECTOT'])
        plt.show()
        # dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])
        # dataframe.set_index(['DATE'], drop=True, inplace=True)
        # dataframe.drop(columns=['LAT', 'LON'], inplace=True)

