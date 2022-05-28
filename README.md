# Crime Free

__A violence detection module that can inform authorities about the details of individuals present at the location during the incident using face recognition.__

## Tech Stack Used:
- Django 
- MySQL
- Tensorflow
- Python 3.7
- OpenCV
- face_recognition
- HTML/CSS/JS
- pywhatkit
- yagmail

## How to set up the application?
- Clone the GitHub repository
``` sh
$ git clone https://github.com/S-JZ/CFEngage.git
```
 - Change directory to crime_free
``` sh
$ cd CFEngage/crime_free
```
- Set up the virtual environment

``` sh
$ virtualenv venv
$ source venv/Scripts/activate
```

- Install dependencies from requirements.txt
``` sh
$ pip install -r requirements.txt
```

- Set up the Django Server
``` sh
$ python manage.py makemigrations
$ python manage.py migrate
$ python manage.py runserver
```

- Open your web-browser and type in ```localhost:8000/```. 

### _Note: Before testing the application_
- You might want to create a super user for the admin page in Django to add people in the database manually.
```sh
$ python manage.py createsuperuser
```
- Enter your desired username, email id and password for the same.
```sh
$ python manage.py runserver
```
- Open your web-browser and type in ```localhost:8000/admin/```. You will be asked to login to the dashboard. Login with the above credentials.
- Now add in the information about the entries in the faces_db folder in the Persons model from the interface manually. By default, there are two images in the faces_db folder.
- Create the dummy Aadhar number, Name and Address of these two persons using the interface.
- You are now set to test the app!



## Environment Variables:
The code uses sensitive information like passwords and secret keys which have been stored in the environment variable for security concerns. These can be re-produced on your system by defining the following in a ```.env``` file in the crime_free subdirectory:
``` sh
DJANGO_SECRET_KEY 
DEBUG              #set to True
DB_NAME            # name of the database (for e.g. dummy_faces)
DB_USER            # database username
DB_PSW             # database password
DATABASE_URL       # localhost
DEVELOPMENT_MODE   #set to True
email   #gmail id for sending mail to authorities
psw     #password of gmail account
phone   #police whatsapp number 
```

## Demo Video Link:


## Design Documentation


## Weekly Reports: Sprints
The file contains the weekly targets and milestones that I worked on throughout the month of May 2022 for the project. These weekly sprints include researching about different tech-stacks, implementing code, reviewing design and testing the application in each iteration.
https://drive.google.com/file/d/1bYgDNUZsmWq3-vn1qwqlIk9FR0pBzR5B/view?usp=sharing

## Future Scope:
1. Reviewing the design to enable faster processing using multiple threads and process synchronization -> parallel processing.
2. Automating database connection with face database.

## Dataset used for Violence Detection:
The dataset used for model creation and preprocessing the video was RWF2000-Video-Database for Violence Detection which contains 2000 clips and 300,000 frames recorded under surveillance camera.
```
@inproceedings{cheng2021rwf,
  title={Rwf-2000: An open large scale video database for violence detection},
  author={Cheng, Ming and Cai, Kunjing and Li, Ming},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={4183--4190},
  year={2021},
  organization={IEEE}
}
```
