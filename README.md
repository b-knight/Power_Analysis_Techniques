# Enabling Better Modeling by Scaling Power Analysis
## Benjamin S. Knight
### October 12th 2019
*How Instacart estimates appropriate sample sizes for A/B test configurations beyond standard t-tests.*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Imagine that one day you are giving a presentation on the results from a recent A/B test (let's say that you changed the color of a particular button in an app from hexcode #33cc33 to hexcode #47d147). To guard against any inadvertant regression, you made sure to perform a difference of means [t-test](https://en.wikipedia.org/wiki/Student%27s_t-test) for your primary counter-metric. In the course of your presentation, you confidently announce that while the point estimate for this counter metric is pointing in an undesired direction, the estimate is statistically insignificant and so is likely noise.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; At this point, your product manager asks if the difference in the counter metric being  statistically insignificant is due to it being *actual* noise, or if the A/B test simply hasn't yet attained sufficient statistical power. Fortunately, you conducted a [power analysis](https://en.wikipedia.org/wiki/Power_(statistics)) before launching your experiment, and while there is a non-zero chance that the regression in the counter metric is real, the results from your power analysis suggest that such an scenario is highly unlikley. Satisfied, your product manager approves the adoption of hexcode #47d147 and the whole team heads out to happy hour to celebrate.
<div>
<div align="center">
<p>*        *        *</p>
</div>
</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; As much as the above scene may reflect the ideal scenario, data analysts and data scientists appreciate that reality is rarely so accomodating. Specifically, while power analysis is more or less a solved problem when one is using conventional techniques such as &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  (there are a variety of on-line power calculators to choose among). However, if we have need for less-conventional methods - say, in the event that our data is not sampled independently - then the the matter of estimating statistical power and suitable sample size quickly becomes non-trivial. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; At Instacart, we often have to make decisions using data that is clustered by customer, retailer, etc., and while there are a variety of tools (e.g. [hierarchical models](https://en.wikipedia.org/wiki/Multilevel_model), and [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)
with robust clustered standard errors) that allow us to overcome these problems, we are still left with the dilemma of how to determine how large a sample is required for these tools to perform well. To this end, running simulations can yield the most accurate estimates. However, simulations are often time-consuming and costly. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Below we outline two *scalable* options for power analysis in the context of clustered data: (1.) approximating the power curve by sampling the exponential distribution, and (2.) estimating the effect size with the variance of residuals. Which of these approaches is preferable will be contingent on one's preferences in regards to the precision / compute trade-off.<br>

## Approximating the Power Curve
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A t-test
