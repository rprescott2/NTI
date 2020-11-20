from rest_framework import serializers

from NTI_srv_neyronka import models as neyron_models


class WindTurbineListSerializer(serializers.ModelSerializer):
    class Meta:
        model = neyron_models.WindTurbine
        fields = ['id', 'model', 'power']


class WindTurbineDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = neyron_models.WindTurbine
        fields = ['id', 'model', 'power']

    def create(self, validated_data):
        return super(WindTurbineDetailSerializer, self).create(validated_data)

    def update(self, instance, validated_data):
        return super(WindTurbineDetailSerializer, self).update(instance, validated_data)


class SolarPanelListSerializer(serializers.ModelSerializer):
    class Meta:
        model = neyron_models.WindTurbine
        fields = ['id', 'model', 'power']


class SolarPanelDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = neyron_models.SolarPanel
        fields = ['id', 'model', 'power']

    def create(self, validated_data):
        return super(SolarPanelDetailSerializer, self).create(validated_data)

    def update(self, instance, validated_data):
        return super(SolarPanelDetailSerializer, self).update(instance, validated_data)


class BuildingTypeListSerializer(serializers.ModelSerializer):
    class Meta:
        model = neyron_models.BuildingType
        fields = ['id', 'name', 'slug']


class BuildingListSerializer(serializers.ModelSerializer):
    class Meta:
        model = neyron_models.BuildingType
        fields = ['id', 'name', 'type', 'consumption']


class BuildingDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = neyron_models.Building
        fields = ['id', 'name', 'type', 'consumption']

    def create(self, validated_data):
        return super(BuildingDetailSerializer, self).create(validated_data)

    def update(self, instance, validated_data):
        return super(BuildingDetailSerializer, self).update(instance, validated_data)


class MeteoDataListSerializer(serializers.ModelSerializer):
    class Meta:
        model = neyron_models.MeteoData
        fields = ['id','created', 'prectot', 'qv2m', 'ps', 't2m']


class MeteoDataDetailSerializer(serializers.ModelSerializer):
    created = serializers.SerializerMethodField(read_only=True)

    def get_created(self, data):
        return data.created

    class Meta:
        model = neyron_models.MeteoData
        fields = ['id', 'created', 'prectot', 'qv2m', 'ps', 't2m']

    def create(self, validated_data):
        return super(MeteoDataDetailSerializer, self).create(validated_data)

    def update(self, instance, validated_data):
        return super(MeteoDataDetailSerializer, self).update(instance, validated_data)