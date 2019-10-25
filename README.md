# Better A/B Testing Through Scalable Power Analysis
### Benjamin S. Knight
#### October 12th 2019
*How Instacart estimates appropriate sample sizes for A/B test configurations beyond standard t-tests.*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Imagine that one day you are giving a presentation on the results from a recent A/B test (let's say that you changed the color of a particular button in an app from hexcode #33cc33 to hexcode #47d147). To guard against any inadvertant regression, you made sure to perform a difference of means [t-test](https://en.wikipedia.org/wiki/Student%27s_t-test) for your primary counter-metric. In the course of your presentation, you confidently announce that while the point estimate for this counter metric is pointing in an undesired direction, the estimate is statistically insignificant and so is likely noise.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; At this point, your product manager asks if the difference in the counter metric being  statistically insignificant is due to it being *actual* noise, or if the A/B test simply hasn't yet attained sufficient statistical power. Fortunately, you conducted a [power analysis](https://en.wikipedia.org/wiki/Power_(statistics)) before launching your experiment, and while there is a non-zero chance that the regression in the counter metric is real, the results from your power analysis suggest that such an scenario is highly unlikley. Satisfied, your product manager approves the adoption of hexcode #47d147 and the whole team heads out to happy hour to celebrate.
<div>
<div align="center">
<p>*        *        *</p>
</div>
</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; As much as the above scene may reflect the ideal scenario, analysts appreciate that reality is rarely so accomodating. While power analysis is more or less a solved problem when one is using conventional difference of means/proportions tests (there are a variety of statistical packages and on-line power calculators to choose among), if we need to employ multivariate modeling - say, in the event that our data is not sampled independently - then the the issue of accurately estimating statistical power and suitable sample size quickly becomes non-trivial. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; At Instacart, we often have to make decisions using data that is clustered by customer, retailer, etc., and while there are a variety of tools (e.g. multilevel models, ordinary least squares with robust clustered standard errors, etc.) that allow us to overcome these problems, we are still left with the dilemma of how to determine how large a sample is required for these tools to perform well. To this end, running simulations can yield the most accurate estimates. However, simulations are often time-consuming and costly. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Below we outline two scalable options for power analysis in the context of clustered data: (1.) approximating the power curve by sampling the exponential distribution, and (2.) estimating the effect size using the variance of residuals. Which of these approaches is preferable will be contingent on one's preferences in regards to the precision / compute trade-off.<br>

### Approximating the Power Curve
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Our goal is to determine the relationship between the number of observations in our sample and the probability of correctly rejecting the NULL hypothesis in the event of a genuine change in the metric of interest. We assume that this relationship between statistical power and sample size is represented by an exponential distribution parameterized by &#955; where the smaller the value of lambda, the greater the increase in probability of detecting the difference for each additional observation added to our sample.<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To illustrate, let's say that we want our A/B test to be able to detect a difference of size 0.01 with 0.70 probability. Four candidate power curves are outlined in Figure 1 in green, blue, red, and purple. The problem is that the value of &#955; is unknown, and so we do not know which of these curves best captures the true relationship between sample size and statistical power.<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Because there is no formula on hand to calculate &#955;, we need to use simulations to estimate it. As the domain of &#955; is [0, &#8734;], one option might be to implement a binary search in the domain of [0, &#8734;]. Unfortunately, for every iteration we need to run a large number of simulations in order to ensure that our initial result was not a fluke, making such an approach impractical. Rather, it makes more sense to make an educated guess. We do so by taking a sampling of simulation results, and using those points to infer the value of &#955;.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Imagine that we surveyed all of the previous experiments that we have run and found that a typical experiment concluded with half a million observations, experiments on the shorter side tended to have 100,000 observations, and experiments on the longer side tended to end after attaining 900,000 observations. With these three trial sample sizes, we conduct 3,000 simulations - 1,000 for each of the three sample sizes. Figure 2 illustrates what this might look like, with the sample size (100k, 500k, and 900k) plotted on the X-axis and the proportion of simulations that sucessfully detected the 0.01 simulated effect (p-value < 0.05) plotted on the Y-axis.<br>
<div>
<div align="center">
<p align="center"><b>Figure 1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure 2</b></p>
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/power_curve_estimation/approximating_the_power_curve_image_1.png" align="middle" width="422" height="216" />
 <img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/power_curve_estimation/approximating_the_power_curve_image_2.png" align="middle" width="422" height="216" />
</div>
</div>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; At this point we still don't know the value of &#955;. However, we do have three points that are closely aligned to the true power curve of interest (in this case, the blue curve). We can use these three points to plot a curve using Scipy's curve fit functionality (Figure 3). Now we have an approximation of the curve and can use it to inform our power analysis. Our goal is to specify an appropriate sample size that will yield 0.70 statical power. The curve we just drew suggests that a sample size of 540,000 will fulfill this requirement, so we implement a final round of 1,000 simulations. The results from these simulations are shown in Figure 4 and suggest that we overshot our mark. The actual amount of statistical power a sample of this size will tend to yield is closer to 0.74. At this point we can either accept the current target sample size recommendation as somewhat excessive, or conduct additional simulations to find a value of n more closely aligned with our specifications. 
<div>
<div align="center">
 <p align="center"><b>Figure 3 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure 4</b></p>
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/power_curve_estimation/approximating_the_power_curve_image_3.png" align="middle" width="422" height="216" />
 <img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/power_curve_estimation/approximating_the_power_curve_image_4.png" align="middle" width="422" height="216" />
</div>
</div>

#### Optimizing Curve Inference
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In the example above we employed a total of 4,000 hypothetical simulations - 3,000 to infer the broad countour of the power curve, and an addition 1,000 to verify the predicted statistical power. The fact that our estimate was slightly off begs the question of whether there are more effective approaches to deploying these simulations. One approach might be to provide Scipy with additional points. If we were able to achieve a somewhat decent approximation with three points, then it reckons that we should be able to achieve a better fit by providing additional information. However, let's say that we have a fair number of (impatient) users needing to launch experiments and that 4,000 simulations is the maximum amount of compute we can feasibly expend for a single power analysis. In this scenario, providing Scipy with six points as opposed to three points implies that we can only use 500 versus 1,000 simulations per point. We will necessarily incur a loss of precision. If we imagine that are on a fishing expedition for the 'true' shape of the power curve, then simulating more hypothetical sample sizes would be the equivalent of using a wider net, whereas using more simulations per point would be comparable to using a finer mesh - increasing both can only improve the likelihood of capturing our quarry, but we are still left with the issue of how best to deploy scarce resources.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Creating some actual data allows us to demonstrate this trade-off more concretely. Let's say that we are interested in assessing the impact of some program designed to increase customer spend. Each of our simulated one million customers has a base order amount that is normally distributed with a mean of $100 and a standard deviation of $25. These customers are randomly assigned to a treatment or control group and in the event of the former, their mean order amount is then scaled by a factor of 1.01 (our treatment effect is a 1% relative increase). In addition, these customers can choose to shop at 1,000 simulated retailers. Some retailers are pricier than others, so we capture this by assigning each retailer a random scalar from a normal distribution of mean 1.0 and a standard deviation of 0.05. In addition, let's assume that order volume is exponentially distributed, with a select number of customers placing orders with far greater frequency than the median customer. Lastly, we introduce some additional noise that is normally distributed (mean of $10, standard deviation of $1).<br>
<div align="center"> 
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/power_curve_estimation/Part_I_Distribution_Specification.png" align="middle" width="588" height="208" />
</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Now that we have specified the data generating process, let's assess different approaches to inferring the power curve (our target power is 0.8). Typically, A/B tests of this natures consisted of 10,000 observations, but have historically varied in their sample sizes with a standard deviation of 2,000 observations. Our first attempt (Method I) elects three candidate sample sizes of 8,000, 10,000, and 12,000 observations. For each of the three candidate sample sizes we draw a random sample of the corresponding size and then estimate the coefficient for the treatment effect using ordinary least squares using Python's Statsmodels package: <br><br>

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
<br>
We repeat this process for 100 times for each point, and then feed each point's mean value to Scipy. Due to the random nature of the sample, there will be occasions when the estimated points do not increase monotonically, causing Scipy's curve fit to fail. This problem is mitigated when we use more than 100 simulations, but for this exercise we just draw a fresh random sample. Once Scipy has approximated the equation of the curve, we extract the final sample size recommendation and verify the results with a final 500 simulations. Our estimated effective power is the proportion of those 500 simulations in which we found the coefficient estimate of the treatment variable to be significant at &#945; < 0.05. Because an individual trial can always be a fluke, we repeat the process described 30 times to build a reasonable distribution.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; From here we tweak the number of points in our implementation using +/- 1 SD (n = 8k, 10k, 12k), +/- 1,2 SDs (n = 6k, 8k, 10k, 12k, 14k), and +/- 1,2,3 SDs (n = 4k, 6k, 8k, 10k, 12k, 14k, 16k). We also also explore estimating each of these points with 100, 200, or 300 simulations each (the verification step is held fixed at 500 simulations). All told, we use a total of 9 models (405,000 simulations) the results of which are displayed in the boxplots below.<br><br>

<div align="center"> 
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/power_curve_estimation/nine_model_results.png" align="middle" width="994" height="478" />
</div>
<br><br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The results suggest that Scipy yields much more accurate curves as more points are added, and that while improving the quality of these points will improved fit the benefit is much less pronounced. For example, in Method VII we deployed 175% of the compute used in Method I (1,400 versus 800 simulations), and yet the the variance in the effective power estimates scarcely decreased. In contrast, Methods V and VII performed very differently despite using a comparable amount of compute (1,500 and 1,400 simulations respectively). Across 30 iterations Methods V and VII yielded sample size estimates that on average, exceeded our target of 0.8 by 0.033 and 0.025 respectively. However, Method V was far more consistent than model VII. For Method VII, the effective power estimates yielded from the recommended sample sizes had a standard deviation of 0.084. For Method V, this figure is only 0.052. These results imply that 32% of the time that we look to Method V for a recommended sample size, we risk oversampling to the extent that our effective power is more than 11 percentage points greater than necessary, or undersampling to the extent that our effective power is more than 6 percentage points lower than specified.<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The results suggest that experimenting with a wide variety of candidate sample sizes




The good news is that this underlying problem is akin to the exploration versus exploitation problem which confronts anyone seeking the ideal configuration of hyper-parameters for optimizing a machine learning model. Known solutions include [Thompson Sampling](https://arxiv.org/pdf/1706.01825.pdf "Parallel and Distributed Thompson Sampling for
Large-scale Accelerated Exploration of Chemical Space") and Bayesian inference.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 




### Approximating Effect Size by Modeling the Variance of Residuals
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; While the approach to power analysis discussed above aims to reduce the amount of simulations required by making these simulations as targeted as possible, the second approach discards the use of simulations entirely. The advantage is that our second approach is drastically faster. However, the disadvantage is that the experimenter is left without a precise estimate of the statistical power actually at their disposal.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this strategy, we use the standard formula for sample size as if we were implementing a difference of means t-test. The key step is estimating an appropriate effect size for the sample size calculation. To this end, we use a series of regression outputs - assuming that the true effect size is baked into the variance of the resulting residuals. Before getting into details, let's review how required sample size is calculated in the context of a t-test.<br><br>

<div align="center">
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/variance_of_residuals/formula_for_ttest_sample_size.png" align="middle" width="434" height="76" />
</div><br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In the formula above, the required sample size - n - is a function of the desired level of statistical significance and the desired amount of statistical power normalized by the effect size. In the numerator, &#945; is the probability of a Type I error (false rejection of the NULL hypothesus, i.e. a false positive) and &#946; is the probability of a Type II error (i.e. a false negative). Observe how if we want to make our test more rigorous and reduce the false positive rate, the the left side of the numerator increases and the resulting recommended sample size also increases. Similarly, if we want to reduce the probability for failing to detect the difference across groups, then we will need to increase the right-hand side of the numerator - also increasing the recommended sample size. In this way the numerator captures what we want the test to achieve. However, our statistical ambitions are constrained by the denominator. The denominator, our effect size, is the ratio between the hypothesized difference and the underlying variation in the metric of interest, a.k.a. our signal-to-noise ratio.<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; When determining the effect size, the hypothesized difference is assumed by the experimenter (hopefully using their domain knowledge). However, the variation - &#963; - is problematic. If we were actually using a standard t-test, we could acquire a decent estimate for &#963; just by looking at how the metric has varied historically. Unfortunately in the world of multivariate regression the variance that we need to normalize is a function of nth dimensional space. There is no query that will fetch us this particular statistic, and we have seen how time-intensive it can be to measure this variation empirically.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The good news is that we have an excellent proxy for the variation that we are interested in - our model's error terms. In t-test land, the noise that we are concermed with is a function of the variable that we are measuring - Y. In the world of multivariate modeling, the variation of interest is the variation in Y NOT captured by our predictor variables. When our model fails to predict a portion of Y's variation, that residual variation is dumped into the &#949; term - also known as the residual! As our model becomes better and as the proportion of Y's variation that in unexplained decreases, our &#949; term shrinks in size.<br>

<div align="center">
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/variance_of_residuals/relation_to_epsilon.png" align="middle" width="452" height="206" />
</div><br>


