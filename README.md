# Enabling Better Modeling by Scaling Power Analysis
## Benjamin S.Knight
### October 12th 2019
*How Instacart estimates appropriate sample sizes for A/B test configurations beyond standard t-tests.*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Imagine that one day you are giving a presentation on the results from a recent A/B test (let's say that you changed the color of a particular button in an app from hexcode #33cc33 to hexcode #47d147). To guard against any inadvertant regression, you made sure to perform a difference of means t-test for your primary counter-metric. In the course of your presentation, you confidently announce that while the point estimate for this counter metric is pointing in an undesired direction, the estimate is statistically insignificant and so is likely noise.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; At this point, your product manager asks if the difference in the counter metric being  statistically insignificant is due to it being actual noise, or if the A/B test simply hasn't yet attained sufficient statistical power. Fortunately, you conducted a [power analysis](https://en.wikipedia.org/wiki/Power_(statistics)) before launching your experiment, and while there is a non-zero chance that the regression in the counter metric is real, the results from your power analysis suggest that such an scenario is highly unlikley. Satisfied, your product manager approves the adoption of hexcode #47d147 and the whole team heads out to happy hour to celebrate.
<div>
<div align="center">
<p>*   *   *</p>
</div>
</div>

