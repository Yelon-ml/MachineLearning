Autos data for the OLS regression model.

The data set contains 205 records with 26 features: 25 dependent, and the predictable one. The task is to predict price of a car using OLS regression model. In the kernel we start with cleaning and processing the data, two approaches are presented:

   1. The first approach considers taking important features due to correlation's plot - we are trying to avoid collinearity between selected variables.
   2. In the second one, we consider influence and significance of features basing on the parameters returned by the first model.

Once features are chosen, we check if the model, run on prepared data, satisfies assumptions required for linear regression model. These conditions are verified using statistical tests, the results are discussed then.
