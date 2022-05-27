from django.db import models

# name : str
# address : str
# Aadhar number : int

class Person(models.Model):
    aadhar_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100)
    address = models.TextField()


