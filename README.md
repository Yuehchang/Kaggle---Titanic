## Kaggle-Titanic (Kaggle practice or competition)
## Titanic project
  ### 1. Variable Definition
   * pclass: ticket class / 1 = first class ...
   * sibsp:  # of siblings or spouses aboard the Titanic
   * parch: # of parents or children aboard the Titanic
   * embarked: Port of embarkation
  ### 2. Features (determine the  categorical or numerical value for visualization purpose)
   * Cat: Survived / Sex / Embarked | Ord: Pclass
	* Con: Age / Fare | Discrete: SibSp / Parch
  ### 3. Analyze the data
   * Which feature are mixed data types? Ticket is a mix of numeric and alphanumeric values | Cabin is alphanumeric.
   * Which feature contain blank, null or empty values? 
      1. Train: Cabin(687) > Age (177) > Embarked(2) 
      2. Test:  Cabin(327) > Age(86)
   * Assumption based on data analysis:
      1. Correlating
        * Which feature contributes significantly to the survival? 
        * Which features correlates to each other?
      2. Completing
        * Age => need to fill NA (using local median)
        * Embarked => need to fill NA (using highest frequency value in dataset)
      3. Correcting
        * Tickets => drop the features due to contain 22% of duplicate values and it may not be correlated to our survivals.
        * Name => drop may not contribute to the survival
        * Passenger Id => drop with the same reason as Name
        * Cabin => drop as it is highly incomplete and contains many null value in both training and test data
      4. Creating
        * Age to age band for using bins, and same do the fare
        * Total family members which combine SibSp and Parch
      5. Classifying (add assumption based on the problem statement earlier)
        * Female had a better chance to survive
        * Age (< Certain age) were more likely to have survived
        * Upper class passengers(Pclass = 1) had a better chance
  ### 4. Data Wrangling
  * Correcting the data by dropping features(good starting point for processing speed) => Drop ticket and Cabin which were highly incomplete in our dataset 
  * Converting categorical value to numeric for modeling purpose => Male to 0, Female to 1
  * Inputting missing value: SEX
	  1. Step 1. Create a np.zeros matrix as tmp storage for median value 
		2. Step 2. Use for loop to find out the median for each combination (Sex =0 & Pclass=1, Sex=0 & Pclass=2, etc)
		3. Step 3. input the M in the matrix
		4. Last step. Slice the dataframe to Series and assign the value 
  * Create Age band / Create Fare band / Create new Feature combining existing feature
 ### 5. Building Models
 * Logistic Regression
 * Perceptron
 * Support Vector Machines
 * Naive Bayes classifier
 * Decision Tree
 * Random Forrest
 * Gradient Boosting
 * Final result: Logistic Regression had a better performacne with **0.82** accurancy in training and **0.78** in test dataset.
 ### 6. Next step / improvement
 * Implemented cross-validation and hyperparameters
 * Based on the accuracy for each model, the trees family are more likely to overfitting due to the high variance problems.
 * Back to data preprocessing part to create dummy variable instead of converting categorical variable to ordinal variable.
 * For the model evaluation, need to implement ROC curve(plot), F1 score, Precision, and Recall scoring metrics in the next version.
 (Accuracy score is usually misleading by the imbalanced data; therefore, F1 score is a better metric to evaluate the performance.)

 





  


        


      
    

