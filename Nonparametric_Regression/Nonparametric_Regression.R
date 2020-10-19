# Read data
data_set <- read.csv("hw04_data_set.csv",header=TRUE)

# Get x and y values
x <- data_set$x
y <- data_set$y
x_train <- x[1:100]
y_train <- y[1:100]
x_test  <- x[101:133]
y_test  <- y[101:133]
remove(x,y)

# introduce bins
bin_width <- 3
origin <- 0
N <- length(y_train)
N_test <- length(y_test)
origin <- 0
maximum_value <- 60
left_borders <- seq(from = origin, to = maximum_value - bin_width, by = bin_width)
right_borders <- seq(from = origin + bin_width, to = maximum_value, by = bin_width)
point_colors <- c("red", "green", "blue")

# Regressogram
p_head_regressogram <- sapply(1:length(left_borders), function(b) 
  {sum((left_borders[b] < x_train & x_train <= right_borders[b])*y_train)/
    sum(left_borders[b] < x_train & x_train <= right_borders[b])})

plot(x_train[1:100], y_train[1:100], type = "p", pch = 19, col = "blue", las = 1,
     xlab = "x", ylab = "y", main = "Regressogram")
points(x_test[1:33], y_test[1:33], type = "p", pch = 19, col = "red")
for (b in 1:length(left_borders)) {
  lines(c(left_borders[b], right_borders[b]), c(p_head_regressogram[b], p_head_regressogram[b]), lwd = 2, col = "black")
  if (b < length(left_borders)) {
    lines(c(right_borders[b], right_borders[b]), c(p_head_regressogram[b], p_head_regressogram[b + 1]), lwd = 2, col = "black") 
  }
}

# Regressogram: RMSE Calculation
y_hat <- sapply (1:length(y_test), function(a) 
          { p_head_regressogram[which(left_borders < x_test[a] & x_test[a] <= right_borders)]})
RMSE_Regressogram <- sqrt(sum((y_test-y_hat)^2)/N_test)

# Running Mean Smoother
data_interval <- seq(from = origin, to = maximum_value, by = 0.01)
p_head_running <- sapply(1:length(data_interval), function(x) 
  {sum((-0.5 < (data_interval[x] - x_train)/bin_width & (data_interval[x] - x_train)/bin_width <= 0.5)*y_train)/
    sum((-0.5 < (data_interval[x] - x_train)/bin_width & (data_interval[x] - x_train)/bin_width <= 0.5))})

# Plot the data
plot(x_train[1:100], y_train[1:100], type = "p", pch = 19, col = "blue", las = 1,
     xlab = "x", ylab = "y", main = "Running Mean Smoother")
points(x_test[1:33], y_test[1:33], type = "p", pch = 19, col = "red")
lines(data_interval, p_head_running, type = "l", lwd = 2, col = "black")

# Running Mean Smoother: RMSE Calculation
y_hat <- sapply (1:length(y_test), function(a) 
{ p_head_running[x_test[a]*100]})
RMSE_Running <- sqrt(sum((y_test-y_hat)^2)/N_test)


# Kernel Smoother
data_interval <- seq(from = origin, to = maximum_value, by = 0.01)
bin_width2 <- 1
p_head_kernel <- sapply(1:length(data_interval), function(x) {
  sum(1 / sqrt(2 * pi) * exp(-0.5 * (data_interval[x] - x_train)^2 / bin_width2^2)*y_train)/
  sum(1 / sqrt(2 * pi) * exp(-0.5 * (data_interval[x] - x_train)^2 / bin_width2^2))})

# Plot the data
plot(x_train[1:100], y_train[1:100], type = "p", pch = 19, col = "blue", las = 1,
     xlab = "x", ylab = "y", main = "Kernel Smoother")
points(x_test[1:33], y_test[1:33], type = "p", pch = 19, col = "red")
lines(data_interval, p_head_kernel, type = "l", lwd = 2, col = "black")

# Kernel Smoother: RMSE Calculation
y_hat <- sapply (1:length(y_test), function(a) 
{ p_head_kernel[x_test[a]*100]})
RMSE_Kernel <- sqrt(sum((y_test-y_hat)^2)/N_test)

# Print RMSEs
cat("Regressogram => RMSE is ", RMSE_Regressogram," when h is ",bin_width)
cat("Running Mean => RMSE is ", RMSE_Running," when h is ",bin_width)
cat("Kernel Smoother => RMSE is ", RMSE_Kernel," when h is ",bin_width2)
