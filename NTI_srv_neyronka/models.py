from django.db import models
from django.db.models.signals import pre_save
from django.dispatch import receiver
from django.utils import timezone


class SolarPanel(models.Model):
    model = models.CharField(verbose_name='Название модели', max_length=155)
    power = models.DecimalField(verbose_name='Мощность', max_digits=15, decimal_places=7)
    square = models.DecimalField(verbose_name='Площадь', blank=True, null=True, max_digits=15, decimal_places=5)
    efficiency = models.DecimalField(verbose_name='КПД', blank=True, null=True, max_digits=12, decimal_places=7)
    Ko = models.DecimalField(verbose_name='Ko', blank=True, null=True, max_digits=12, decimal_places=7)
    U = models.DecimalField(verbose_name='U', blank=True, null=True, max_digits=12, decimal_places=7)
    class Meta:
        verbose_name = 'Солнечная панель'
        verbose_name_plural = 'Солнечные панели'

    def __str__(self):
        return self.model

    def __unicode__(self):
        return self.model


class WindTurbine(models.Model):
    model = models.CharField(verbose_name='Название модели', max_length=155)
    power = models.DecimalField(verbose_name='Мощность', max_digits=15, decimal_places=7)
    square = models.DecimalField(verbose_name='Площадь', blank=True, null=True, max_digits=15, decimal_places=5)
    efficiency = models.DecimalField(verbose_name='КПД', blank=True, null=True, max_digits=12, decimal_places=7)
    diameter = models.DecimalField(verbose_name='Диаметр', blank=True, null=True, max_digits=12, decimal_places=7)
    Q = models.DecimalField(verbose_name='Q', blank=True, null=True, max_digits=12, decimal_places=7)
    Ng = models.DecimalField(verbose_name='Ng', blank=True, null=True, max_digits=12, decimal_places=7)

    class Meta:
        verbose_name = 'Ветряк'
        verbose_name_plural = 'Ветряки'

    def __str__(self):
        return self.model

    def __unicode__(self):
        return self.model


class BuildingType(models.Model):
    name = models.CharField('Название группы', max_length=155)
    slug = models.SlugField('Метка', max_length=10)

    class Meta:
        verbose_name = 'Группа зданий'
        verbose_name_plural = 'Группы зданий'

    def __str__(self):
        return self.name

    def __unicode__(self):
        return self.name


class Building(models.Model):
    type = models.ForeignKey(BuildingType, verbose_name='Группа здания', on_delete=models.CASCADE)
    name = models.CharField('Название строения', max_length=50)
    consumption = models.DecimalField(verbose_name='Расход здания на этаж', max_digits=15, decimal_places=7)

    class Meta:
        verbose_name = 'Здание'
        verbose_name_plural = 'Здания'

    def __str__(self):
        return self.name

    def __unicode__(self):
        return self.name


class Settings(models.Model):
    name = models.CharField('Название', max_length=155)
    std = models.CharField(verbose_name='std', max_length=155)
    mean = models.CharField(verbose_name='mean', max_length=155)
    max_y = models.CharField(verbose_name='max_y', max_length=1551)
    min_y = models.CharField(verbose_name='min_y', max_length=155)

    class Meta:
        verbose_name = 'Настройка'
        verbose_name_plural = 'Настройки'

    def __str__(self):
        return self.name

    def __unicode__(self):
        return self.name


class MeteoData(models.Model):
    created = models.DateTimeField(default=timezone.now)
    prectot = models.DecimalField(verbose_name='Кол-во осадков', max_digits=16, decimal_places=8)
    qv2m = models.DecimalField(verbose_name='Влажность', max_digits=16, decimal_places=8)
    ps = models.DecimalField(verbose_name='Давление', max_digits=16, decimal_places=8)
    t2m = models.DecimalField(verbose_name='Температура', max_digits=16, decimal_places=8)

    class Meta:
        verbose_name = 'Метеоданные'
        verbose_name_plural = 'Метеоданные'


class ActualData(models.Model):
    created = models.DateTimeField(default=timezone.now)
    prectot = models.DecimalField(verbose_name='Кол-во осадков', max_digits=16, decimal_places=8)
    qv2m = models.DecimalField(verbose_name='Влажность', max_digits=16, decimal_places=8)
    ps = models.DecimalField(verbose_name='Давление', max_digits=16, decimal_places=8)
    t2m = models.DecimalField(verbose_name='Температура', max_digits=16, decimal_places=8)

    class Meta:
        verbose_name = 'Актуальные данные'
        verbose_name_plural = 'Актуальные данные'


@receiver(pre_save, sender=ActualData)
def ActualData_pre_save(sender, instance=None, **kwargs):
    ActualData.objects.all().delete()