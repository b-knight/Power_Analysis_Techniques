# Better A/B Testing Through Scalable Power Analysis
### Benjamin S. Knight
#### October 12th 2019
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Controlling for covariates allows us to detect smaller effect sizes with less data [(Deng, Xu, Kohavi, & Walker, 2013)](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf). However, we are still left with the question of how much data constitutes 'enough'. Power analysis for such multivariate modeling can be tricky, as there are no simple formulae we can use to estimate an adequate sample size. At Instacart, we use two primary strategies to engage this problem: (1.) interpolating the shape of the power curve by deploying simulations in a strategic fashion, and (2.) running comparable models and inferring the unknown effect size by estimating the variance of residuals. These approaches are not mutually exclusive, and may yield the best results when used in tandem. Which approach (or combination of approaches) is most appropriate will be a function of the confidence with which one can simulate the underlying data, the cost one is likely to pay in the event that the required sample size is overestimated, as well as practical considerations regarding the amount of available compute.*<br> 
___
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Imagine that one day you are giving a presentation on the results from a recent A/B test (let's say that you changed the color of a particular button in an app from hexcode #33cc33 to hexcode #47d147). To guard against any inadvertant regression, you made sure to perform a difference of means [t-test](https://en.wikipedia.org/wiki/Student%27s_t-test) for your primary counter-metric. In the course of your presentation, you mention that while the counter-metric is pointing in an undesired direction, the difference is statistically insignificant and is likely not a genuine cause for concern.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; At this point, your product manager asks if the lack of statistical significance is due to the regression being *actual* noise, or if the A/B test simply hasn't yet attained sufficient statistical power. Fortunately, you conducted a [power analysis](https://en.wikipedia.org/wiki/Power_(statistics)) before launching your experiment, and while there is a non-zero chance that the regression in the counter metric is real, the results from your power analysis suggest that such an scenario is highly unlikley. Satisfied, your product manager approves the adoption of hexcode #47d147 and the whole team heads out to happy hour to celebrate.<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; While the above scenario is admittedly facetious, the danger that arises when statistical analysis is conducted absent a power analysis is very real. By '[power](https://en.wikipedia.org/wiki/Power_(statistics)),' we are referring to the probability that we successfully detect the difference across groups when such a difference genuinely exists. When we say that an A/B test is adequately 'powered,' we are effectively saying that we have given reality ample opportunity to reveal itself. Without adequately sizing how large a sample our A/B test requires, we run a very real risk of committing a 
[Type-II error](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors).<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; While power analysis is more or less a solved problem when one is using difference of means/proportions tests (there are a variety of statistical packages and on-line power calculators to choose among), if we need to employ [multivariate modeling](https://en.wikipedia.org/wiki/Multivariate_statistics) - say, in the event that we want to control for covariates or our data is not sampled independently - then the issue of accurately estimating statistical power and consequently, a suitable sample size quickly becomes non-trivial. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For instance, at Instacart we often have to make decisions using data that is clustered by customer, retailer, etc., and while there are a variety of tools (e.g. multilevel models, ordinary least squares with [robust clustered standard errors](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html), etc.) that allow us to overcome these problems, we are still left with the dilemma of how to determine how large a sample is required for these tools to perform well. To this end, running simulations can yield the most accurate estimates. However, simulations are often time-consuming and costly.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Below we outline two scalable options for power analysis in the context of multivariate analysis: (1.) interpolating the power curve by sampling the exponential distribution via simulation, and (2.) estimating the effect size using the variance of residuals. Which of these approaches is preferable will be contingent on one's confidence that the underlying data generating process can be accurately captured via simulation, as well as one's preferences in regards to the precision / compute trade-off.<br>

### Interpolating the Power Curve
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Our ultimate goal is to determine the relationship between the number of observations in our sample, and the probability of correctly rejecting the NULL hypothesis in the event of a genuine change in the metric of interest. We assume that this relationship between statistical power and sample size is represented by the cumulative distribution function for the exponential distribution parameterized by &#955; where the smaller the value of lambda, the greater the increase in probability of detecting the difference for each additional observation added to our sample.
<div align="center"> 
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_I_power_curve_estimation/lambda.png" align="middle" width="573" height="111" />
</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To illustrate, let's say that we want our A/B test to be able to detect a difference of size 0.01 with 0.70 probability. Four candidate power curves are outlined in Figure 1 in green, blue, red, and purple. The problem is that the value of &#955; is unknown, and so we do not know which of these curves best captures the true relationship between sample size and the probability of us detecting the hypothetical effect.<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Because there is no formula on hand to calculate &#955;, we need to use simulations to estimate it. A single simulation per candidate sample size is too susceptible to the idiosyncrasies of the individual sample. One could sample one hundred observations, and from them glean strong evidence against the NULL hypothesis. A different set of one hundred observations might suggests a very different conclusion. If we are going to hazard a guess that (for example) n = 100 represents a suitable sample size, then it is advisable to run hundreds, if not thousands of simulations - a considerable investment in both time and compute. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; So long as we are using simulations, it makes more sense to be strategic regarding where we deploy them. One way of doing so is by taking a sampling of simulation results, and using those points to infer the value of &#955;. Imagine that we surveyed all of the previous experiments that we have run and found that a typical experiment concluded with half a million observations, experiments with run times on the shorter side tended to have 100,000 observations, and experiments on the longer side tended to end after attaining 900,000 observations. With these three trial sample sizes, we conduct 3,000 simulations - 1,000 for each of the three sample sizes. Figure 2 illustrates what this might look like, with the sample sizes (100k, 500k, and 900k) plotted on the X-axis and the proportion of simulations that sucessfully detected the 0.01 simulated effect (p-value < 0.05) plotted on the Y-axis.<br><br>
<div>
<div align="center">
<p align="center"><b>Figure 1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure 2</b></p>
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_I_power_curve_estimation/approximating_the_power_curve_image_1.png" align="middle" width="422" height="216" />
 <img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_I_power_curve_estimation/approximating_the_power_curve_image_2.png" align="middle" width="422" height="216" />
</div>
</div>
<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; At this point, we still don't know the value of &#955;. However, we do have three points that are closely aligned to the true power curve of interest (in this case, the blue curve). We can use these three points to plot a curve using Scipy's [Curve Fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) functionality (Figure 3). Now we have an approximation of the curve and can use it to inform our power analysis. Our goal is to specify an appropriate sample size that will yield 0.70 statical power. The curve we just drew suggests that a sample size of 540,000 will fulfill this requirement, so we implement a final round of 1,000 simulations. The results from these simulations are shown in Figure 4 and suggest that we overshot our mark. The actual amount of statistical power a sample of this size will tend to yield is closer to 0.74. At this point we can either accept the current target sample size recommendation as somewhat excessive, or conduct additional simulations to find a value of n more closely aligned with our specifications. 

<div>
<div align="center">
 <p align="center"><b>Figure 3 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure 4</b></p>
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_I_power_curve_estimation/approximating_the_power_curve_image_3.png" align="middle" width="422" height="216" />
 <img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_I_power_curve_estimation/approximating_the_power_curve_image_4.png" align="middle" width="422" height="216" />
</div>
</div>

#### Optimizing Curve Inference
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In the example above we employed a total of 4,000 hypothetical simulations - 3,000 to infer the broad countour of the power curve, and an addition 1,000 to verify the predicted statistical power. The fact that our estimate was slightly off begs the question of whether there are more effective approaches to deploying these simulations. One approach might be to provide Scipy with additional points. If we were able to achieve a somewhat decent approximation with three points, then it reckons that we should be able to achieve a better fit by providing additional information. However, let's say that we have a fair number of (impatient) users needing to launch experiments and that 4,000 simulations is the maximum amount of compute we can feasibly expend for a single power analysis. In this scenario, providing Scipy with six points as opposed to three points implies that we can only use 500 versus 1,000 simulations per point. We will necessarily incur a loss of precision. If we imagine that are on a fishing expedition for the 'true' shape of the power curve, then simulating more hypothetical sample sizes would be the equivalent of using a wider net, whereas using more simulations per point would be comparable to using a finer mesh - increasing both can only improve the likelihood of capturing our quarry, but we are still left with the issue of how best to deploy scarce resources.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Creating some actual data allows us to demonstrate this trade-off more concretely. Let's say that we are interested in assessing the impact of some program designed to increase customer spend per order. We simulate one million customers, each having a a base order amount that is normally distributed with a mean of $100 and a standard deviation of $25. These customers are randomly assigned to a treatment or control group and in the event of the former, their mean order amount is then scaled by a factor of 1.01 (our treatment effect is a 1% relative increase). In addition, these customers can choose to shop at 1,000 simulated retailers. Some retailers are pricier than others, so we capture this by assigning each retailer a random scalar from a normal distribution of mean 1.0 and a standard deviation of 0.05. In the grocery business, day-of-week seasonality matters. If I place my order on a weekend, there is a high chance that I'm stocking up on enough groceries to last the week. An order placed on a Wednesday will tend to be smaller by comparison. To reflect this, let's add some linearly distributed scalars in the range [0.7,1.3] (i.e. Monday orders are scaled to 70% whereas Sunday orders are scaled to 130%). Let's assume that order volume is exponentially distributed, with a select number of customers placing orders with far greater frequency than the median customer. Lastly, we introduce some additional noise that is normally distributed (mean of $10, standard deviation of $1).<br>
<div align="center"> 
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_I_power_curve_estimation/DGP.png" align="middle" width="571" height="367" />
</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Now that we have specified the data generating process, let's assess different approaches to inferring the power curve (our target power is 0.8). Let's say that previous A/B tests of this nature consisted of 40,000 observations, but have historically varied in their sample sizes with a standard deviation of 10,000 observations. Our first attempt (Method I) elects three candidate sample sizes of 30,000, 40,000, and 50,000 observations. For each of the three candidate sample sizes we draw a random sample of the corresponding size and then estimate the coefficient for the treatment effect using ordinary least squares using Python's Statsmodels package. Because order amounts are clustered on customers (Customer_ID), we need to instruct Statsmodels to use robust clustered standard errors: <br><br>

```python
import statsmodels.api as sm

results = []
model = sm.OLS(Y, X).fit(method='pinv').get_robustcov_results(
        'cluster', groups = df.Customer_ID, 
         use_correction=True, df_correction=True)
if model.pvalues[1] < 0.05:
    results.append(1)    
else:
    results.append(0)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We repeat this process for 100 times for each point, and then feed each point's mean value to Scipy. Due to the random nature of the sample, there will be occasions when the estimated points do not increase monotonically, causing Curve Fit to fail. This problem occurs less and less frequently the more simulations per point we use, but for this exercise we just draw a fresh random sample. Once we have converted the list of p-values into an estimate of statistical power, we cast our candidate sample sizes and power estimates as dual arrays and infer the exponentual CDF using the Python below. 

```python
import numpy as np
from scipy.optimize import curve_fit

desired_power = 0.8

def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

eta = []
for i in results:
    eta.append(i)
eta = np.asarray(eta)

cdf = []
for i in points:
    cdf.append(i)
cdf = np.asarray(cdf)

popt, pcov = curve_fit(exp_func, eta, cdf)
recommended_n = int(popt[0]*np.exp(-(popt[1]) * (desired_power)) + popt[2])
   ```
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Once Curve Fit has approximated the equation of the curve, we extract the final sample size recommendation and verify the results with a final set of 500 simulations. Our estimated effective power is the proportion of those 500 simulations in which we found the coefficient estimate of the treatment variable to be significant at &#945; < 0.05. Because an individual trial can always be a fluke, we repeat the entire process 100 times to build a reasonable distribution.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; From here we tweak the number of points in our implementation using +/- 1 SD (n = 30k, 40k, 50k), +/- 1,2 SDs (n = 20k, 30k, 40k, 50k, 60k), and +/- 1,2,3 SDs (n = 10k, 20k, 30k, 40k, 50k, 60k, 70k). We also explore estimating each of these points with 100, 200, or 300 simulations each (the verification step is held fixed at 500 simulations). All told, we use a total of 9 methods (405,000 simulations) the results of which are displayed in the boxplots and table below.<br><br>

<div align="center"> 
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_I_power_curve_estimation/nine_model_results_v2.png" align="middle" width="994" height="415" />
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_I_power_curve_estimation/nine_model_table_v2.png" align="middle" width="852" height="321" />
</div>
<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The above results convey a broad pattern wherein Curve Fit yields more accurate curves as more points are added. Of greater interest is the fact that the extent of this improvement tends to be far greater than that attained by increasing the number of simulations per point. For example, in Method VII we deployed 300% of the compute used in Method I (900 versus 300 simulations). Our reward for tripling our compute was to reduce by 40% the variance in the effective power estimates (6 percentage points worth of statistical power in Method VII versus 10 percentage points in Method I). In contrast, Methods V and VII performed very differently despite using a comparable amount of compute (1,000 and 900 simulations respectively). Across 100 iterations Methods V and VII yielded sample size estimates that on average, exceeded our target of 80% by 2 percentage points and 4 percentage points respectively. Method V was far also far more consistent than model VII. For Method VII, the effective power estimates yielded from the recommended sample sizes had a standard deviation of 6.1 percentage points. For Method V, this figure is roughly a third as large at 2.4 percentage points. The implication is that the rewards are larger if we prioritize breadth versus depth.<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The above results also strike home how point estimation with fewer than 200 iterations is problematic (the various outliers shown above are particularly disconcerting). For instance, if we can take the results from Method I and subtract the effective power as estimated using 500 simulations from our desired power if 80%. The standard error of these 100 differences is of the order of 10 percentage points. This implies that approximately 32% of the time, the recommended sample size will be off by *at least* this amount. Depending on the context, this could imply millions of unnecessary observations and/or weeks of needless additional experiment run time.

### Approximating Effect Size by Modeling the Variance of Residuals
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; While the previous approach to power analysis aims to reduce the number of simulations required by making these simulations as targeted as possible, the second approach discards the use of simulations entirely. The advantage is that our second approach is drastically faster. However, the disadvantage is that the experimenter is left without a precise estimate of the statistical power actually at their disposal. As we show, when one uses this approach while simultaneously controlling for covariates, the tendency is to overestimate the required sample. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this strategy, we use the standard formula for sample size as if we were implementing a difference of means t-test. The key step is estimating an appropriate effect size for the sample size calculation. To this end, we use a series of regression outputs - assuming that the true effect size is baked into the variance of the resulting residuals. Before getting into details, let's review how required sample size is calculated in the context of a t-test.<br><br>

<div align="center">
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_2_variance_of_residuals/formula_for_ttest_sample_size.png" align="middle" width="434" height="76" />
</div><br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In the formula above, the required sample size (n) is a function of the desired level of statistical significance and the desired amount of statistical power normalized by the effect size. In the numerator, &#945; is the probability of a Type I error (false rejection of the NULL hypothesus, i.e. a false positive) and &#946; is the probability of a Type II error (i.e. a false negative). If we want to make our test more rigorous and reduce the false positive rate, then we increase the value of the left side of the numerator which induces an increase in the recommended sample size. Similarly, if we want to reduce the probability that we fail to detect an actual difference across groups (i.e. the NULL hypothesis is false), then we will need to increase the right-hand side of the numerator. In this way, the numerator captures what we want from the test and how stringent we want to be in to achieving it. Meanwhile the denominator is the ratio between the hypothesized difference (the treatment mean and the control mean) and the underlying variation in the metric of interest, a.k.a. our signal-to-noise ratio.<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;It is the estimation of this effect size where we run into problems. When determining the effect size, the hypothesized difference is assumed by the experimenter (hopefully using their domain knowledge). However, the variation denoted by the denominator - &#963; - is problematic. If we were actually using a standard t-test, we could acquire a decent estimate for &#963; just by looking at how the metric has varied historically. Unfortunately in the world of multivariate regression, the variance that we need to normalize is a function of nth dimensional space. There is no query that will fetch us this particular statistic, and we have seen how time-intensive it can be to measure this variation empirically. Recall our model from earlier:
<div align="center">
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_2_variance_of_residuals/Specification_v2.png" align="middle" width="383" height="47" />
</div><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We stymied are in that we do not know the true value of &#963; At the same time, we can think of each of the predictor variables above as carving out a portion of &#963; - the more predictor variables we have, the larger the share of &#963; that is accounted for. It is the unaccounted for portion of &#963; that dampens our ability to detect the treatment effect's influence on y. The good news is that this noise is captured in the term &#949;. Even better, we can easily estimate &#949; by using the model to make some predictions (y&#770;), and taking the differences between those predictions and the true values of y. Our model will fail to predict y with perfect precision, but the unaccounted for variation - the residual - is dumped into the &#949; term. The better our understanding of how &#949; behaves, the better our understandig of the noise our model is up against and the greater the precision with which we can designate an appropriate sample size.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; With our goal being to better understand the process which generated these residuals, we run another model - this time regressing the residuals on a constant term. The standard error estimate generated by this second model is our approximation of the remaining variation unaccounted for by our original model. All that remains is to account for sample size. The sample size with which we run our model drastically influences the size of the estimated standard error. We want our estimate of the residual variation to be independent of the sample size we used to estimate it. We undo this influence by multiplying the standard error by the squareroot of the sample size that was used. When we began our power analysis, we already had in mind our best guess for the difference we expect to manifest between the control and treated groups. All that remains is to plug our approximation of &#963; into the denominator and we have our estimate of the signal-to-noise ratio that our A/B test must account for once we launch our experiment.<br>       
<div align="center">
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_2_variance_of_residuals/residuals_v2.png" align="middle" width="751" height="215" />
</div>

```python
    analysis_df = analysis_df[['Order_Amt', 'Customer_ID', 'Mean_Order_Amt', 
                               'Mean_Retailer_Order_Amt','Mean_DOW_Order_Amt']]
    
    X = analysis_df[['Mean_Order_Amt', 'Mean_Retailer_Order_Amt','Mean_DOW_Order_Amt']]
    X = sm.add_constant(X)
    Y = analysis_df[['Order_Amt']]
    residuals_df = sm.OLS(Y.astype(float), X.astype(float)).fit()
    
    X2 = analysis_df[['Customer_ID']]
    X2['Residual'] = residuals_df.resid
    X2['Constant'] = 1
    clustered_res = sm.OLS(X2['Residual'], X2['Constant']).fit(method='pinv'). \
                       get_robustcov_results('cluster', groups = X2['Customer_ID'], 
                       use_correction=True, df_correction=True)
    
    clustered_sd = clustered_res.bse[0] * np.sqrt(analysis_df.shape[0])
    effect_size = absolute_mde / clustered_sd
    recommended_n = int(sm.stats.tt_ind_solve_power(effect_size = effect_size, 
                        alpha = rejection_region, power = desired_power, 
                        alternative = 'larger'))
    return recommended_n
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To assess how our approach work in practice, let's use the data described above in which the value of an order is a function of the customer's underlying mean order amount (prior to treatment), the mean order amount for that retailer (prior to treatment), and the day of the week, and whether or not the customer recieved the treatment. To see how change in the size of the treatment effect influences the accuracy of our sample size estimates, let's use a variety of effect sizes - 5%, 2.5%, 1%, and 0.5% relative (or $4.67, $2.36, $1.00, and $0.42 worth of absolute order valuation relative to a mean value of $125.28). Residual variation is the primary input for our sample size estimate, so we should examine a variety of specifications as well. We define Model I was the full model outlined above. For each of the model II's, III, and IV, we remove an additional covariate (the lagged customer mean amount, the lagged retailer mean amount, and lastly, the day of the week). All told, we have 16 frameworks. Each framework yields a sample size recommendation. To assess the effective power of that sample size within that framework, we run the appropriate model 500 times and record the proportion of outcomes in which we evaluated the coefficient estimate of the treatment variable as statistically significant at the p-value < 0.05 threshold. Our estimate of effective power in-hand, we repeat the process 100 times to build a distribution of outcomes for that framework. The results of these 800,000 simulations are shown below.<br><br>

<div align="center">
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_2_variance_of_residuals/sixteen_model_results.png" align="middle" width="948" height="396" />
</div>
<div align="center">
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_2_variance_of_residuals/results_table_v2.png" align="middle" width="1000" height=330" />
</div>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; With the exception of the 16th and final framework, the results fall into two broad categories. Frameworks 5 through 15 all overestimated the required sample size. 


<div align="center">
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/part_2_variance_of_residuals/multicollinearity_impact_v2.png" align="middle" width="724" height="349" />
</div><br>

### Bringing it All Together




