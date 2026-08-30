# Frame the problem and look at the big picture  
1. [X] *Define the objective in business terms.*

Predict, using only publicly available data, whether a Falcon 9 first stage will be successfully reused.

2. [x] *How will your solution be used?* 

The prediction will be used for estimating a competitive launch price.

3. [x] *What are the current solutions/workarounds (if any)?*

The cost of a Falcon 9 launch depends mainly on payload mass and orbit type. Given a launch mission, SpaceX 
determines the feasibility of landing by considering several factors:
  - Propellant margin
  - Landing profile
  - Thermal and Aerodynamic constraints
  - Dynamic fleet management and booster history
  - In-flight real-time feasibility 
  
Furthermore, SpaceX uses advanced multi-variable physics simulations, real-time trajectory optimization software, 
and strict margins for remaining fuel in order to ensure that the energy needed to deliver the payload leaves
enough leftover propellant to safely guide the booster through extreme atmospheric environments back to a standtill.

This current method is costly and time-consuming, but it is certainly worth it. The decisions revolving such missions
are critical and leaves a very small margin of error. The goal of creating this machine learning model is not to
compete against this current solution, but to complement it by providing quick estimates during the early phases
of the project.
   
4. [x] *How should you frame this problem (supervised/unsupervised, online/offline, etc.)*

This is a **supervised learning task**, since the model can be trained with labeled examples. It is a classification task
since the model will be asked to predict whether a launch is successful or not. In particular, this is a **binary
classification** problem, since the prediction is one of two values: "Success" or "Failure". Finally, data is gathered
from previous SpaceX missions and is small enough to fit in memory. Furthermore, rocket launches are scheduled and 
intermittent, hence there is no continuous flow of data coming into the system. So, plain **batch learning** is implemented.

5. [x] _How should performance be measured?_

Average Precision is used for model selection and hyperparameter tuning, while Precsion is used for final evaluation.

6. [x] _Is the performance measure aligned with the business objective?_

The objective requires that the number of false positives by the model be minimized. a false positive means that a 
failed landing was flagged as successful. In this case, the company would price low for the launch, only to find out that
the mission requires an expendable booster. The company would then be contractually locked to deliver below the actual
cost of the expendable launch. On the other hand, a false negative in this case means that a successful 
booster recovery was predicted by the model as a landing failure. If such an error occurs, then the mission must consider
an expendable booster, leading us to overprice and decrease the chance of winning the bid. The loss in a false negative
comes when a rival underbids the company. This is more forgiving than the loss from a false positive error.

As a competitor, we consider a false positve error to carry the highest commercial risk, and therefore the least ideal
scenario given the objective of the project. Hence, Precision was chosen as the performance measure. Precision will be used in
the final evaluation of the best model. For model selection and hyperparameter tuning, average precision is used.

7. [x] _What would be the minimum performance needed to reach the business objective?_

Given that the publicly available data is highly imbalanced in favor of successful missions, we choose a minimum of 20%
relative improvement from the baseline Average Precision. For deployment, we will establish an operating point at the
inflection point of the Precision-Recall curve, maximizing Precision just before Recall suffers a severe drop.

# Get the data   
_Note: automate as much as possible so you can easily get fresh data._  

1. [X] _List the data you need and how much you need._

The following is the list of general mission attributes that are needed:
- **Booster Version** - `str`; The rocket engine used for the launch, e.g., Falcon 9.
- **Payload Mass** - `float`; The total mass (in kg) of cargo carried by the spacecraft during the launch, e.g., 800 kg
- **Orbit** - `str`; The type of orbit in which the spacecraft is placed upon launch, e.g., Sun-synchronous orbit (SSO), Low Earth orbit (LEO).
- **LaunchSite** - `str`; name of the launch site, e.g., VAFB SLC 4E
- **Landing Success** - `bool`; status of the landing
- **Landing Type** - `str`; type of landing pad, e.g., ASDS. 
- **Flights** - `int`; number of flights with the core used in a launch
- **Reused** - `bool`; whether or not the spacecraft is reused for other flights.
- **LandingPad** - `str`; name of the landing pad, e.g., LZ-2
- **Block** - `int`; version of the rocket, e.g., Falcon 9 block 5
- **ReusedCount** - `int`; number of times the rocket is reused.
- **Serial** - `str`; identification of the booster used, e.g., B1051
- **Longitude** - `float`; longitude of the launchpad.
- **Latitude** - `float`; latitude of the launchpad.
- **Mission Type** - `str`; e.g., communications, dedicated rideshare.

In addition to the general attributes, we will also consider **proximity to major highway, coastline, and railway**
derived from a geospatial analysis of the launch sites. Distances to nearest major highway, coastline, and railway 
will be considered as features over launch site name and longitude and latitude as they are more interpretable and 
generalizable to new locations.

2. [X] _Find and document where you can get that data._

- General attributes - [Launch Library 2 API (LL2)](https://thespacedevs.com/llapi)
- Payload Mass - [Jonathan McDowell's General Catalogue of Artificial Space Objects (GCAT)](https://planet4589.org/space/gcat/web/launch/ldir.html)
- US Highways Geodata - [usroads.json by bricedev](https://gist.github.com/bricedev/96d2113bd29f60780223?short_path=650495d)
- US Railways Geodata - [North American Rail Network Lines dataset](https://geodata.bts.gov/datasets/usdot::north-american-rail-network-lines/about)
- Global Coastlines Geodata - [Coastline data from Natural Earth](https://www.naturalearthdata.com/downloads/10m-physical-vectors/)
- Florida Coastlines Geodata - [Florida Shoreline dataset](https://hub.arcgis.com/datasets/myfwc::florida-shoreline-1-to-12000-scale/about)

3. [X] _Check how much space it will take._

The data to be obtained will take at most 1 GB of memory. This is small enough to be downloaded on a local computer.

4. [X] _Check legal obligations, and get the authorization if necessary._

All documents are publicly available. Hence, authorization or any license is not required.

5. [X] _Create a workspace (with enough storage space)._  
6. [X] _Get the data._

Data obtained from Launch Library 2 API and GCAT are considered raw data, and are cached in `data/raw` folder,
whereas geodata are considered external data and are cached in `data/external` folder.

7. [X] _Convert the data to a format you can easily manipulate (without changing the data itself)._

Raw data from LL2 and GCAT, as well as external data, are cached in `data/raw` or `data/external` folders. The relevant
attributes were then extracted from the raw data and are saved as a `.csv` file to the `data/interim` folder. The csv file
will be easy to access for the data exploration later.

8. [X] _Check the size and type of data (time series, sample, geographical, etc.)._

Data is small (552 instances) by machine learning standards, and each instance represents a Falcon 9 launch mission.
All attributes are strings except for payload mass, longitude and latitude.

9. [X] _Sample a test set, put it aside, and never look at it (no data snooping!)._

Given the small dataset, a stratified split based on `class` was required.

# Explore the data  
Note: try to get insights from a field expert for these steps.  

1. Create a copy of the data for exploration (sampling it down to a manageable size if necessary).
2. Create a Jupyter notebook to keep record of your data exploration.  
3. Study each attribute and its characteristics:  
    - Name  
    - Type (categorical, int/float, bounded/unbounded, text, structured, etc.)
    - % of missing values  
    - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.)
    - Possibly useful for the task?  
    - Type of distribution (Gaussian, uniform, logarithmic, etc.)
4. For supervised learning tasks, identify the target attribute(s).
5. Visualize the data.  
6. Study the correlations between attributes.  
7. Study how you would solve the problem manually.  
8. Identify the promising transformations you may want to apply.  
9. Identify extra data that would be useful (go back to "Get the Data" on page 502).  
10. Document what you have learned.  

# Prepare the data  
Notes:    
- Work on copies of the data (keep the original dataset intact).  
- Write functions for all data transformations you apply, for five reasons:  
    - So you can easily prepare the data the next time you get a fresh dataset  
    - So you can apply these transformations in future projects  
    - To clean and prepare the test set  
    - To clean and prepare new data instances  
    - To make it easy to treat your preparation choices as hyperparameters  

1. Data cleaning:  
    - Fix or remove outliers (optional).  
    - Fill in missing values (e.g., with zero, mean, median...) or drop their rows (or columns).  
2. Feature selection (optional):  
    - Drop the attributes that provide no useful information for the task.  
3. Feature engineering, where appropriate:  
    - Discretize continuous features.  
    - Decompose features (e.g., categorical, date/time, etc.).  
    - Add promising transformations of features (e.g., log(x), sqrt(x), x^2, etc.).
    - Aggregate features into promising new features.  
4. Feature scaling: standardize or normalize features.  

# Short-list promising models  
Notes: 
- If the data is huge, you may want to sample smaller training sets so you can train many different models in a 
reasonable time (be aware that this penalizes complex models such as large neural nets or Random Forests).  
- Once again, try to automate these steps as much as possible.    

1. Train many quick and dirty models from different categories (e.g., linear, naive, Bayes, SVM, Random Forests, 
neural net, etc.) using standard parameters.  
2. Measure and compare their performance.  
    - For each model, use N-fold cross-validation and compute the mean and standard deviation of their performance. 
3. Analyze the most significant variables for each algorithm.  
4. Analyze the types of errors the models make.  
    - What data would a human have used to avoid these errors?  
5. Have a quick round of feature selection and engineering.  
6. Have one or two more quick iterations of the five previous steps.  
7. Short-list the top three to five most promising models, preferring models that make different types of errors.  

# Fine-Tune the System  
Notes:  
- You will want to use as much data as possible for this step, especially as you move toward the end of fine-tuning.   
- As always automate what you can.    

1. Fine-tune the hyperparameters using cross-validation.  
    - Treat your data transformation choices as hyperparameters, especially when you are not sure about them (e.g., 
   should I replace missing values with zero or the median value? Or just drop the rows?).  
    - Unless there are very few hyperparameter values to explore, prefer random search over grid search. If training 
   is very long, you may prefer a Bayesian optimization approach (e.g., using a Gaussian process priors, as described 
   by Jasper Snoek, Hugo Larochelle, and Ryan Adams ([https://goo.gl/PEFfGr](https://goo.gl/PEFfGr)))  
2. Try Ensemble methods. Combining your best models will often perform better than running them individually.  
3. Once you are confident about your final model, measure its performance on the test set to estimate the generalization error.

> Don't tweak your model after measuring the generalization error: you would just start overfitting the test set.  
  
# Present your solution  
1. Document what you have done.  
2. Create a nice presentation.  
    - Make sure you highlight the big picture first.  
3. Explain why your solution achieves the business objective.  
4. Don't forget to present interesting points you noticed along the way.  
    - Describe what worked and what did not.  
    - List your assumptions and your system's limitations.  
5. Ensure your key findings are communicated through beautiful visualizations or easy-to-remember statements (e.g., 
"the median income is the number-one predictor of housing prices").  

# Launch!  
1. Get your solution ready for production (plug into production data inputs, write unit tests, etc.).  
2. Write monitoring code to check your system's live performance at regular intervals and trigger alerts when it drops.  
    - Beware of slow degradation too: models tend to "rot" as data evolves.   
    - Measuring performance may require a human pipeline (e.g., via a crowdsourcing service).  
    - Also monitor your inputs' quality (e.g., a malfunctioning sensor sending random values, or another team's output 
   becoming stale). This is particularly important for online learning systems.  
3. Retrain your models on a regular basis on fresh data (automate as much as possible).  