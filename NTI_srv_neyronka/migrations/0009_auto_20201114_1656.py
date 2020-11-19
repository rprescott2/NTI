# Generated by Django 2.0.13 on 2020-11-14 16:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('NTI_srv_neyronka', '0008_auto_20201113_1345'),
    ]

    operations = [
        migrations.AddField(
            model_name='solarpanel',
            name='diameter',
            field=models.DecimalField(blank=True, decimal_places=7, max_digits=12, null=True, verbose_name='Диаметр'),
        ),
        migrations.AddField(
            model_name='solarpanel',
            name='efficiency',
            field=models.DecimalField(blank=True, decimal_places=7, max_digits=12, null=True, verbose_name='КПД'),
        ),
        migrations.AddField(
            model_name='solarpanel',
            name='square',
            field=models.DecimalField(blank=True, decimal_places=5, max_digits=15, null=True, verbose_name='Площадь'),
        ),
        migrations.AddField(
            model_name='windturbine',
            name='diameter',
            field=models.DecimalField(blank=True, decimal_places=7, max_digits=12, null=True, verbose_name='Диаметр'),
        ),
        migrations.AddField(
            model_name='windturbine',
            name='efficiency',
            field=models.DecimalField(blank=True, decimal_places=7, max_digits=12, null=True, verbose_name='КПД'),
        ),
        migrations.AddField(
            model_name='windturbine',
            name='square',
            field=models.DecimalField(blank=True, decimal_places=5, max_digits=15, null=True, verbose_name='Площадь'),
        ),
    ]
