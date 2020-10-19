# below, I tried to implement the multi-class discrimination by regression with sigmoid function that is discussed in the textbook.
# the code is not very self-explaining, but I am sorry that I was out of time. Thank you for your understanding.

#pre-determined parameters
eta <- 0.01
epsilon <- 1e-3

# functions
safelog <- function(x) {
  x[x <= 0] <- 1
  return (log(x))
}
gradient_w <- function(x_train, y_train, y_train_hat) {
  return (-colSums(matrix(y_train - y_train_hat, nrow = nrow(x_train), ncol = ncol(x_train), byrow = FALSE) * x_train))
  }
gradient_w0 <- function(y_truth, y_predicted) {
  return (sum(y_truth - y_predicted) * y_predicted * (1-y_predicted))
  }
sigmoid <- function(X, w, w0) {
  return (1 / (1 + exp(-(X %*% w + w0))))
}

# read and separate the data
data_points <- read.csv("hw02_data_set_images.csv",header=FALSE)
labels_set <- read.csv("hw02_data_Set_labels.csv",header=FALSE)
labels_set$V1 <- gsub("A",1,labels_set$V1)
labels_set$V1 <- gsub("B",2,labels_set$V1)
labels_set$V1 <- gsub("C",3,labels_set$V1)
labels_set$V1 <- gsub("D",4,labels_set$V1)
labels_set$V1 <- gsub("E",5,labels_set$V1)
labels_set$V1 <- as.numeric(labels_set$V1)

data_set <- cbind(data_points, labels_set)

classes <- 5      # because we have 5 letters
take <- 25        # we want to take first 25 rows of each class as training data
skip <- 14        # and the next 14 rows of each class as test data
labels <- array(c(1,2,3,4,5))

training_data <- data_set[rep(0:(classes-1), each=take) * (take+skip) + (1:take),]
test_data <- data_set[take + rep(0:(classes-1),each=skip) * (take+skip) + (1:skip),]


y_train <- data.matrix(training_data[,321])
colnames(y_train) <- NULL
zeros <- matrix(rep(0:0),nrow = 100, ncol =1)
zeros25 <- matrix(rep(0:0),nrow = 25, ncol =1)
zeros50 <- matrix(rep(0:0),nrow = 50, ncol =1)
zeros75 <- matrix(rep(0:0),nrow = 75, ncol =1)
y_train_new = cbind(c(c(y_train[y_train==1]),c(zeros)),
                    c(c(zeros25), c(y_train[y_train==2]), c(zeros75)),
                    c(c(zeros50), c(y_train[y_train==3]),c(zeros50)),
                    c(c(zeros75), c(y_train[y_train==4]),c(zeros25)),
                    c(c(zeros),c(y_train[y_train==5])))

y_train_new[ y_train_new == 2 ] = 1
y_train_new[ y_train_new == 3 ] = 1
y_train_new[ y_train_new == 4 ] = 1
y_train_new[ y_train_new == 5 ] = 1



x_train <- data.matrix(training_data[,1:320])
colnames(x_train) <- NULL
y_test <- data.matrix(test_data[,321])
colnames(y_test) <- NULL
zeros56 <- matrix(rep(0:0),nrow = 56, ncol =1)
zeros14 <- matrix(rep(0:0),nrow = 14, ncol =1)
zeros28 <- matrix(rep(0:0),nrow = 28, ncol =1)
zeros42 <- matrix(rep(0:0),nrow = 42, ncol =1)
y_test_new = cbind(c(c(y_test[y_test==1]),c(zeros56)),
                    c(c(zeros14), c(y_test[y_test==2]), c(zeros42)),
                    c(c(zeros28), c(y_test[y_test==3]),c(zeros28)),
                    c(c(zeros42), c(y_test[y_test==4]),c(zeros14)),
                    c(c(zeros56),c(y_test[y_test==5])))

y_test_new[ y_test_new == 2 ] = 1
y_test_new[ y_test_new == 3 ] = 1
y_test_new[ y_test_new == 4 ] = 1
y_test_new[ y_test_new == 5 ] = 1

x_test <- data.matrix(test_data[,1:320])
colnames(x_test) <- NULL
# initialize w and w0
set.seed(521)
w <- matrix(runif(ncol(x_train)*5, min = -0.01, max = 0.01),nrow = ncol(x_train), ncol = 5)
w0 <- matrix(runif(1, min = -0.01, max = 0.01), nrow = 125, ncol = 5)



# learn w and w0 using gradient descent
iteration <- 1
objective_values <- c()
while (1) {
  y_train_hat <- sigmoid(x_train, w, w0)
  y_train_hat_labels <- y_train_hat
  for (c in 1:125) {
    y_train_hat_labels[c,which.max(y_train_hat_labels[c,])]=1
  }
  y_train_hat_labels[y_train_hat_labels != 1] = 0
  # objective function
  objective_values <- c(objective_values, 0.5*sum(y_train_new * y_train_hat_labels+y_train_new * y_train_hat_labels))
  
  w_old <- w
  w0_old <- w0

  w <- w - eta * gradient_w(x_train, y_train_new, y_train_hat_labels)
  w0 <- w0 - eta * gradient_w0(y_train_new, y_train_hat_labels)
  
  if (iteration == 500) {
    break
  }
  
  iteration <- iteration + 1
}


# plot objective function during iterations
plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")


