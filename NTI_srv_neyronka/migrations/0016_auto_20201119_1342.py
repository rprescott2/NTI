# Generated by Django 3.1.3 on 2020-11-19 13:42

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('NTI_srv_neyronka', '0015_meteodata'),
    ]

    operations = [
        migrations.AlterField(
            model_name='meteodata',
            name='created',
            field=models.DateField(default=django.utils.timezone.now),
        ),
    ]