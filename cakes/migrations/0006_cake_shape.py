# Generated by Django 5.0.1 on 2024-01-08 21:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cakes', '0005_cakecomment'),
    ]

    operations = [
        migrations.AddField(
            model_name='cake',
            name='shape',
            field=models.CharField(null=True),
        ),
    ]
