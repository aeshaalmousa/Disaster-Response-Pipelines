# Disaster-Response-Pipelines

### Installations:
- pandas
- numpy
- sqlalchemy
- nltk
- sklearn
- pickle
- flask
- plotly
### Project Motivation:
In this project, analyzing disaster data from Figure Eight to build a model for an API that classifies disaster messages.
Also, the project include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 
### File Descriptions:
- app
| - template
| |- master.html  main page of web app
| |- go.html   classification result page of web app
|- run.py   Flask file that runs app

- data
|- disaster_categories.csv   data to process 
|- disaster_messages.csv   data to process
|- process_data.py  cleaning code(ETL processe python script)
|- InsertDatabaseName.db    database to save clean data to

- models
|- train_classifier.py  machine learning pipline python script
|- classifier.pkl   saved model 

- README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements:
The data provider is <a href='https://www.figure-eight.com'>figure-eight</a>
<a href='https://www.udacity.com/school-of-data-science'> Udacity </a> for its Instructions and providing a strong learning path.
