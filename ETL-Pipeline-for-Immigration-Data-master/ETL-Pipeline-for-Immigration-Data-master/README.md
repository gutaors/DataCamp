# ETL Pipeline for Immigration Data

## Project Summary
The goal of the project is to develop an ETL pipeline for the immigration data. I plan to extract the data from three different data sources, transform the data using pyspark and pandas, and save the data in the parquet files.

The project follows the follow steps:
* Step 1: Scope the Project and Gather Data
* Step 2: Explore and Assess the Data
* Step 3: Define the Data Model
* Step 4: Run ETL to Model the Data
* Step 5: Complete Project Write Up

### Step 1: Scope the Project and Gather Data

#### Project Scope

I want to examine the origin country and the destination city of the immigrats. I will use the i94 data together with the city demographic data and the country gapminder data. The end solution will be a fact table with the immigration data and two dimention tables with the details of the destination city and the origin country. I'll use both pandas and spark as the tool.

#### Describe and Gather Data
The city demographic data comes from OpenSoft. It contains the demographic information of US cities. The web address of the source is https://public.opendatasoft.com/explore/dataset/us-cities-demographics/export/.

The country data comes from gapminder.org. It contains the fertility, life, population, child mortality, gdp and region of the countries. The web address of the source is https://www.gapminder.org/data/.

The immigration data comes from the US National Tourism and Trade Office. The web address of the source is https://travel.trade.gov/research/reports/i94/historical/2016.html.

### Step 2: Explore and Assess the Data
#### Explore the Data
Identify data quality issues, like missing values, duplicate data, etc.

- The city dataframe has few missing values.
- The Gdp and Child Mortality columns of the country dataframe have missing values, but less than 10%.
- The immigration_data_sample dataframe, assessed in place of the immigration_data, has a few columns ('entdepu', 'occup', 'insnum', 'visapost') with more than 50% of the missing values, which need to be removed.  
- The immigration_data has data type issues. 'i94yr','i94mon','i94bir','biryear', and 'admnum' columns should be integers not doubles. 'i94res', 'i94visa', and 'i94mode' columns should be first converted to integers and then to strings.

### Step 3: Define the Data Model
#### 3.1 Conceptual Data Model

The **fact table** will contain information from the I94 immigration data. It will be saved to parquet files partitioned by the origin country (i94res) and the destination city (i94port).

The **first dimension table** will contain destination city's demographic information from the city dataframe and the immigration_data dataframe. It will be saved to parquet files partitioned by the destination city (i94port).

The **second dimension table** will contain origin country's information from the country dataframe and the immigration_data dataframe. It will be saved to parquet files partitioned by the origin country (i94res).

#### 3.2 Mapping Out Data Pipelines
The steps necessary to pipeline the data into the chosen data model:

- Clean immigration_data and create the fact table immigration_table partitioned by i94port and i94res.
- Clean the city demographic data, create the first dimention table destination_table and write to parquet file partitioned by i94port.
- Clean the country gapminder data, create the second dimention table origin_table and write to parquet file partitioned by i94res.

### Step 4: Run Pipelines to Model the Data
#### 4.1 Create the data model
Build the data pipelines to create the data model.

#### 4.2 Data Quality Checks
There are two data quality checks to ensure the pipeline ran as expected. The first one is a check on the number of data rows, and the second one is a check on the number of data columns. Both checks are to ensure the completeness of data.

#### 4.3 Data dictionary

The **fact table** contains information from the I94 immigration data. The source of the data is the US National Tourism and Trade Office.

- i94yr - 4 digit year
- i94mon - Numeric month
- *i94res* - 3 digit code of origin country
- *i94port* - 3 character code of destination city
- i94addr - state code of destination city
- i94visa - reason for immigration (1 = Business, 2 = Pleasure, 3 = Student)
- i94mode - 1 digit travel code (1 = 'Air', 2 = 'Sea', 3 = 'Land', 9 = 'Not reported')
- i94bir - Age of Respondent in Years
- biryear - year of birth
- gender - 1 character code of gender
- visatype - Class of admission legally admitting the non-immigrant to temporarily stay in U.S.
- airline -  Airline used to arrive in U.S.
- fltno - Flight number of Airline used to arrive in U.S.
- admnum - Admission Number

The **first dimension table** contains destination city's demographic information from the city dataframe and the immigration_data dataframe. The source of the data is OpenSoft and the US National Tourism and Trade Office.

- City - Name of the city
- State_Code - 2 character state code
- *i94port* - 3 character code of destination city
- State - State name
- Median_Age - Median age of the city
- Male_Population - The number of male population
- Female_Population - The number of female population
- Total_Population - The number of total population
- Number_of_Veterans - The number of vaterans
- Foreign-born - The number of foreign-born population
- Average_Household_Size - The average household size
- American_Indian_and_Alaska_Native - The number of american indian and alaska native population
- Asian - The number of asian population
- Black_or_African-American - The number of african american population
- Hispanic_or_Latino - The number of hispanic or latino population
- White - The number of white population

The **second dimension table** contains origin country's information from the country dataframe and the immigration_data dataframe. The source of the data is gapminder.org and the US National Tourism and Trade Office.

- *i94res* - 3 digit code of origin country
- Country - Country name
- Year - 4 digit year
- Fertility - Births per woman
- Life - The average number of years a newborn would live if the current mortality rates were to stay the same
- Population - The total population of the country
- Child_Mortality - Death of children under 5 years of age per 1000 live births
- Gdp - GDP per capita PPP$ inflation adjusted
- Region - The region of the country in the world

#### Step 5: Complete Project Write Up

I chose Spark because it can handle several petabytes of data at a time, and also has developer libraries and APIs which support multiple programming languages including python. Spark's power lies in its ability to combine very different techniques and processes together into a coherent process.

The data can be updated annualy to compare the year to year trend. It can also be updated quaterly or monthly if there are business needs to monitor these trends.

 * If the data was increased by 100x, we can move the data to S3, and use multiple clusters on AWS to process the data. Another option is to use incremental updates using Uber's Hudi.

 * If the data populates a dashboard that must be updated on a daily basis by 7am every day, we can use Airflow, because it has the scheduler tool. The Airflow scheduler monitors all tasks and DAGs, and triggers the task instances whose dependencies have been met. In the scheduler, we can set the SLA to meet the 7am goal.

 * If the database needed to be accessed by 100+ people, we can move the data warehouse to Redshift. There needs to be cost-benefit analysis though. We could also publish the parquet files to HDFS and give people read access.  
