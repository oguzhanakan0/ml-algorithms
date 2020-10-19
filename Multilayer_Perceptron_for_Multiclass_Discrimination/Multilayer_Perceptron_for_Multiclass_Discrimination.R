# read and combine data points and labels
data_points <- read.csv("hw03_data_Set_images.csv",header=FALSE)
labels_set <- read.csv("hw03_data_Set_labels.csv",header=FALSE)
labels_set$V1 <- gsub("A",1,labels_set$V1)
labels_set$V1 <- gsub("B",2,labels_set$V1)
labels_set$V1 <- gsub("C",3,labels_set$V1)
labels_set$V1 <- gsub("D",4,labels_set$V1)
labels_set$V1 <- gsub("E",5,labels_set$V1)
labels_set$V1 <- as.numeric(labels_set$V1)
data_set <- cbind(data_points, labels_set)
colnames(data_set) <- NULL
# separate training and test data
K <- 5            # Because we have 5 classes
take <- 25        # we want to take first 25 rows of each class as training data
skip <- 14        # and the next 14 rows of each class as test data
# labels <- array(c(1,2,3,4,5))
training_data <- data_set[rep(0:(K-1), each=take) * (take+skip) + (1:take),]
test_data <- data_set[take + rep(0:(K-1),each=skip) * (take+skip) + (1:skip),]

# initialize N, H, D, K
D <- 320
H <- 20
N <- 125
N_test <-70
eta <- 0.005
epsilon <- 1e-3
max_iteration <- 200

# initialize train and test data
y_truth <- training_data[,D+1]
y_truth_new <- matrix(0, N, K)
y_truth_new[cbind(1:N, y_truth)] <- 1
x_train <- data.matrix(training_data[,1:D])

y_truth_test <- test_data[,D+1]
y_truth_test_new <- matrix(0, N_test, K)
y_truth_test_new[cbind(1:N_test, y_truth_test)] <- 1
x_test <- data.matrix(test_data[,1:D])

# define safelog, sigmoid, and softmax functions
safelog <- function(x) {
  return (log(x + 1e-100))
}

sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

softmax <- function(Z, V) {
  scores <- exp(cbind(1, Z) %*% V)
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

# randomly initialize W and V
set.seed(521)
W <- matrix(runif((D + 1) * H, min = -0.01, max = 0.01), nrow=D+1, ncol=H)
set.seed(521)
V <- matrix(runif((H + 1) * K, min = -0.01, max = 0.01), nrow=H+1, ncol=K)
objective_values <- c()
iteration <- 1

# first run
Z <- sigmoid(cbind(1, x_train) %*% W)
y_predicted <- softmax(Z, V)
objective_values <- c(objective_values, -sum(y_truth_new * safelog(y_predicted)))

# iterate
while(1){
  V_temp <- matrix(0, H+1, K)
  W_temp <- matrix(0, D+1, H)
  for(i in 1:K){
    V_temp[,i] <- eta* t((y_truth_new[,i] - y_predicted[,i])) %*% cbind(1, Z)
  }
  for(h in 1:H){
    W_temp[,h] <- eta* matrix(rowSums(sapply(X = 1:ncol(y_truth_new), function(i) (y_truth_new[,i] - y_predicted[,i])*V[h+1,i])) * Z[,h] * (1-Z[,h]),nrow=1,ncol=125) %*% cbind(1, x_train)
  }
  V <- V + V_temp
  W <- W + W_temp
  
  Z <- sigmoid(cbind(1, x_train) %*% W)
  y_predicted <- softmax(Z, V)
  objective_values <- c(objective_values, -sum(y_truth_new * safelog(y_predicted)))
  
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  iteration = iteration + 1
}

print(objective_values)
plot(1:iteration, objective_values[1:iteration],
     type = "l", lwd = 2, las = 1)

# calculate confusion matrix
y_predicted <- apply(y_predicted, 1, which.max)
confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)

# run the test data and print its confusion matrix
Z_test <- sigmoid(cbind(1, x_test) %*% W)
y_predicted_test <- softmax(Z_test, V)
y_predicted_test <- apply(y_predicted_test, 1, which.max)
confusion_matrix2 <- table(y_predicted_test, y_truth_test)
print(confusion_matrix2)
