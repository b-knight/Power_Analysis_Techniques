# Enabling Better Modeling by Scaling Power Analysis
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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; As much as the above scene may reflect the ideal scenario, data analysts and data scientists appreciate that reality is rarely so accomodating. While power analysis is more or less a solved problem when one is using conventional difference of means/proportions tests (there are a variety of on-line power calculators to choose among), if we need to employ multivariate modeling - say, in the event that our data is not sampled independently - then the the issue of accurately estimating statistical power and suitable sample size quickly becomes non-trivial. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; At Instacart, we often have to make decisions using data that is clustered by customer, retailer, etc., and while there are a variety of tools (e.g. 
[multilevel_models](https://en.wikipedia.org/wiki/Multilevel_model), and 
[ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)
with robust clustered standard errors) that allow us to overcome these problems, we are still left with the dilemma of how to determine how large a sample is required for these tools to perform well. To this end, running simulations can yield the most accurate estimates. However, simulations are often time-consuming and costly. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Below we outline two *scalable* options for power analysis in the context of clustered data: (1.) approximating the power curve by sampling the exponential distribution, and (2.) estimating the effect size with the variance of residuals. Which of these approaches is preferable will be contingent on one's preferences in regards to the precision / compute trade-off.<br>

### Approximating the Power Curve
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Our goal is to determine the relationship between the number of observations in our sample and the probability of correctly rejecting the NULL hypothesis in the event of a genuine change in the metric of interest. We assume that this relationship between statistical power and sample size is represented by an exponential distribution parameterized by &#955; where the smaller the value of lambda, the greater the increase in probability of detecting the difference for each additional observation added to our sample.<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To illustrate, let's say that we want our A/B test to be able to detect a difference of size 0.01 with 0.70 probability. Four candidate power curves are outlined in Figure 1 in green, blue, red, and purple. The problem is that the value of &#955; is unknown, and so we do not know which of these curves best captures the true relationship between sample size and statistical power.<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Because there is no formula on hand to calculate &#955;, we need to use simulations to estimate it. As the domain of &#955; is [0, &#8734;], one option might be to implement a binary search in the domain of [0, &#8734;]. Unfortunately, for every iteration we need to run a large number of simulations in order to ensure that our initial result was not a fluke, making such an approach impractical. Rather, it makes more sense to make an educated guess. We do so by taking a sampling of simulation results, and using those points to infer the value of &#955;.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We can see this in Figure 2. Imagine that we surveyed all of the previous experiments that we have run and found that a typical experiment concluded with half a million observations, experiments on the shorter side tended to conclude with 200,000 observations, and experiments on the longer side tended to end after attaining 800,000 observations. With these three trial values, we conduct 3,000 simulations - 1,000 for each of the three sample sizes. The sample size (2k, 5k, and 8k) is plotted on the X-axis. On the Y-axis we plot the proportion of simulations that sucessfully detected the 0.01 simulated effect (p-value <= 0.05).
<div>
<div align="center">
<p align="center"><b>Figure 1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure 2</b></p>
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/power_curve_estimation/approximating_the_power_curve_image_1.png" align="middle" width="422" height="216" />
 <img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/power_curve_estimation/approximating_the_power_curve_image_2.png" align="middle" width="422" height="216" />
</div>
</div>
dadada
<div>
<div align="center">
 <p align="center"><b>Figure 3 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure 4</b></p>
<img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/power_curve_estimation/approximating_the_power_curve_image_3.png" align="middle" width="422" height="216" />
 <img src="https://github.com/b-knight/Power_Analysis_Techniques/blob/master/power_curve_estimation/approximating_the_power_curve_image_4.png" align="middle" width="422" height="216" />
</div>
</div>

