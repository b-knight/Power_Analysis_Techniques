# Set up some data
x    <- c(1,   2,  3,  4,  5,  6,  7,  8,  9, 10)
y    <- c(11, 48, 29, 29, 95, 24, 91, 69, 36, 73)
data <- data.frame(x, y)

my_lm<- lm(data$y ~ data$x)


PRESS <- function(linear.model) {
  press_residual <- residuals(linear.model)/(1-lm.influence(linear.model)$hat)
  return(press_residual)
}

sd(PRESS(my_lm))
sd(resid(my_lm))
