# Generated by Django 5.0.1 on 2024-01-05 19:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cakes', '0002_cake_created_date_cake_size'),
    ]

    operations = [
        migrations.CreateModel(
            name='CakeFinalJson',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('jsondata', models.JSONField(null=True)),
            ],
        ),
    ]
