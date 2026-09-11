\# Smart Irrigation Water Recommendation System



End-to-end Machine Learning project for predicting recommended irrigation amounts using meteorological and agricultural data.



\## Project Objective



The goal of this project is to build a smart irrigation recommendation system capable of estimating the amount of irrigation water required based on environmental conditions.



The project covers the full Data Science and Machine Learning lifecycle:



\- Data exploration

\- Feature engineering

\- Time-based validation

\- Machine Learning model development

\- Model evaluation

\- API development

\- Docker containerization

\- AWS cloud deployment



\## Dataset



The project uses a scientific irrigation dataset containing historical meteorological and irrigation-related observations from 2000 to 2022.



Main variables include:



\- Air pressure

\- Average temperature

\- Maximum temperature

\- Minimum temperature

\- Precipitation

\- Relative humidity

\- Wind velocity

\- Sunshine duration

\- Reference evapotranspiration (ET0)

\- Water demand



The operational target used in this project is:



`Recommended\_Irrigation\_mm`



Negative irrigation values were converted to zero because a real irrigation system should never recommend a negative irrigation quantity.



\## Exploratory Data Analysis



EDA was performed to study the relationships between weather variables and irrigation demand.



Important observations included:



\- Higher precipitation generally reduces irrigation requirements.

\- ET0 shows a positive relationship with irrigation demand.

\- Water demand is positively associated with recommended irrigation.

\- Temperature and sunshine duration also contribute to irrigation prediction.



\## Feature Engineering



The date variable was transformed into useful temporal features:



\- Month

\- Day of year



The final model uses meteorological, agricultural, and temporal variables.



\## Train-Test Strategy



Because the dataset contains time-series observations, a chronological split was used instead of a random train-test split.



Training period:



2000–2020



Testing period:



2021–2022



This approach reduces the risk of information leakage from future observations.



\## Models Tested



Several regression models were compared:



| Model | MAE | RMSE | R² |

|---|---:|---:|---:|

| Baseline | 1.721 | 2.022 | -0.036 |

| Linear Regression | 0.926 | 1.179 | 0.648 |

| Random Forest | 0.124 | 0.241 | 0.985 |

| Gradient Boosting | 0.173 | 0.245 | 0.985 |

| Tuned Random Forest | 0.123 | 0.242 | 0.985 |



\## Final Model



The final model selected was a Random Forest Regressor.



Final test performance:



\- MAE: 0.124 mm

\- RMSE: 0.241 mm

\- R²: 0.985



The simpler Random Forest configuration was retained because hyperparameter tuning produced only marginal improvement.



\## Temporal Validation



The model was also evaluated using year-based temporal validation.



Average results:



\- MAE: 0.145

\- RMSE: 0.309

\- R²: 0.979



This provides additional evidence that the model generalizes well across different time periods within the dataset.



\## Model Deployment



The trained model was serialized using Joblib.



A FastAPI application was created to expose the prediction model through a REST API.



The API accepts environmental conditions and returns:



`recommended\_irrigation\_mm`



\## Docker



The FastAPI application was containerized using Docker.



Main container components:



\- Python 3.12

\- FastAPI

\- Uvicorn

\- Pandas

\- Scikit-learn

\- Joblib



\## AWS Architecture



The deployment architecture is:



Data Science Model  

→ FastAPI  

→ Docker  

→ Amazon ECR  

→ Amazon EC2  

→ Public REST API



The Docker image is stored in Amazon Elastic Container Registry and deployed on an EC2 instance.



\## API Endpoint



\### POST /predict



Example input:



```json

{

&#x20; "date": "2021-05-01",

&#x20; "air\_pressure\_hpa": 987.62,

&#x20; "average\_temperature\_c": 9.33,

&#x20; "maximum\_temperature\_c": 15.26,

&#x20; "minimum\_temperature\_c": 2.72,

&#x20; "precipitation\_mm": 0,

&#x20; "relative\_humidity\_pct": 57.42,

&#x20; "wind\_velocity\_ms": 2.76,

&#x20; "sunshine\_hours": 6.29,

&#x20; "et0": 3.2,

&#x20; "water\_demand": 1.344

}

