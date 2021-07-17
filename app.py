import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import numpy as np
import pandas as pd
from dash.dependencies import Input,Output


df=pd.read_csv('diabetes.csv')
# Create X (all the feature columns)
x=df.drop("Outcome",axis=1)

# Create y (the target column)
y=df["Outcome"]

# Split the data into training and test sets
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y)

from sklearn.preprocessing import StandardScaler

Scaler=StandardScaler()
Scaler.fit(x_train)
x_train=Scaler.transform(x_train)
x_test=Scaler.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=1)
classifier.fit(x_train,y_train)
# i did feature selection and came out with 4 variables
import joblib

model3=joblib.load('model_save2')

external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']

fig=px.scatter(df,x="BMI",y="Age",
               size="Pregnancies",color="Glucose",
               log_x=True,size_max=60,width=800,height=400,
               title='RELATIONSHIP ANALYSIS WITHIN THE VARIABLES')
fig2=px.bar(df,x="Age",y="DiabetesPedigreeFunction",color="Glucose",barmode="group")
app=dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.layout=html.Div([
    html.Div([
        html.H1(
            children=" DIABETIC ANALYSIS DASHBOARD",
            style={
                'color':'white',
                'backgroundColor':'grey',
            }
        )
    ],className='row'),
    html.Div([
        html.Div([
            html.P(
                'Diabetes is a disease in which the body ability to produce or respond to the hormone insulin is impaired resulting in abnormal metabolism of carbs and elevated levels of glucose in the blood and urine. ')
        ])

    ]),
    html.Div([
        html.Div([
            dcc.Graph(id='relationship with variables',
                      figure=fig),

        ],className='six columns'),
        html.Div([
            dcc.Graph(id='analysis on glucose levels and age',
                      figure=fig2),
        ],className='six columns'),
    ],className='row'),
    html.Div([
        html.Div([
            html.P(
                'From the analysis above on the chart at our right we can conclude that woman who have a body mass index (BMI)of between 20 to 40and also of age 20 to 30 have moderate levels of glucose meaning that their chances of being diabetic are low,and it also shows that woman of old age have high levels of glucose in their blood stream which increases their chances of being diabetic.On the other chart we can see that woman who have high levels of diabetesPedigreeFunction are of age 21 to 34 years .Diabetes Pedigree Function provides history of relatives and genetic relationship of those relatives with patients.Higher HPF means patient is more likely to have diabetes ')
        ])
    ]),
    html.Div([
        html.Div([
            html.H1(children="MACHINE LEARNING COMPONENT",style={
                'color':'white','backgroundColor':'grey'
            })
        ])
    ]),
    html.Div(children=[
        html.H1(children='DIABETES ANALYSIS',style={'textAlign':'center'}),
        html.Div(children=[
            html.Label('ENTER NUMBER OF PREGNANCIES CARRIED'),
            dcc.Input(id='input_1',placeholder='NUMBER OF PREGNANCIES CARRIED',value='number'),


        ]

        ),
        html.Div(children=[
            html.Label('ENTER GLUCOSE LEVEL'),
            dcc.Input(id='input_2',placeholder='GLUCOSE LEVELS',value='number')

        ]

        ),
        html.Div(children=[
            html.Label('ENTER YOUR BODY MASS INDEX'),
            dcc.Input(id='input_3',placeholder='BMI',value='number')
        ])

    ]

    ),
    html.Div(children=[
        html.Label('ENTER YOUR AGE'),
        dcc.Input(id='input_4',placeholder='AGE',value='number'),
        html.H1(id='prediction_result')
    ]),
    html.Div(id="result")
])

app.callback(Output(component_id='result',component_property='children'),
             [Input(component_id='input_1',component_property='value'),
              Input(component_id='input_2',component_property='value'),
              Input(component_id='input_3',component_property='value'),
              Input(component_id='input_4',component_property='value')],
             prevent_initial_call=False

             )


def prediction(value_1,value_2,value_3,value_4):
    input_X = np.array([value_1,
                      value_2,
                      value_3,
                      value_4

                      ]).reshape(1,-1)
    prediction = model3.predict(input_X)[0]
    return "prediction:{}".format(round(prediction,1))



if __name__=='__main__':
    app.run_server(debug=True)
