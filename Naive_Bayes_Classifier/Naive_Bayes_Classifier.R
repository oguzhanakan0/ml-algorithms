# 1. READING AND SEPARATING DATA

# read and combine data points and labels
data_points <- read.csv("hw01_data_Set_images.csv",header=FALSE)
labels_set <- read.csv("hw01_data_Set_labels.csv",header=FALSE)
labels_set$V1 <- gsub("A",1,labels_set$V1)
labels_set$V1 <- gsub("B",2,labels_set$V1)
labels_set$V1 <- gsub("C",3,labels_set$V1)
labels_set$V1 <- gsub("D",4,labels_set$V1)
labels_set$V1 <- gsub("E",5,labels_set$V1)
labels_set$V1 <- as.numeric(labels_set$V1)

data_set <- cbind(data_points, labels_set)

# separate training and test data
classes <- 5      # because we have 5 letters
take <- 25        # we want to take first 25 rows of each class as training data
skip <- 14        # and the next 14 rows of each class as test data
labels <- array(c(1,2,3,4,5))

training_data <- data_set[rep(0:(classes-1), each=take) * (take+skip) + (1:take),]
test_data <- data_set[take + rep(0:(classes-1),each=skip) * (take+skip) + (1:skip),]

y_train <- training_data[,321]
x_train <- training_data[,1:320]

y_test <- test_data[,321]
x_test <- test_data[,1:320]

# 2. ESTIMATING PARAMETERS

# prior probabilities - p(y=c)
probabilities <- sapply(X = 1:classes, FUN = function(c) {length((y_train == labels[c])[(y_train == labels[c])==TRUE])/length(y_train)})
prior <- cbind(labels,probabilities)

# parameter estimation
pcd <- sapply(X=1:5, FUN = function(c){
       sapply(X=1:320, FUN = function(x){
              sum(x_train[c:(c+24)+24*(c-1),x])/length((y_train == labels[c])[(y_train == labels[c])==TRUE])})})

# 3. SCORING FUNCTION 

# function for calculating the scoring according to Bernoulli distribution
# this function takes two matrix, a and b (namely a=data points and b=probability of pixel is black at 
# position d for class c), and applies the logarithmic scoring function of Bernoulli density.
safelog <- function(x) {log(x + 1e-100)}

fun1 <- function(a,b){
        sapply(X=1:classes, FUN = function(c){
          sapply(X=1:nrow(a), FUN = function(x){
            sum(sapply(X=1:length(a), FUN = function(d){
                a[x,d]*safelog(b[d,c]) + (1-a[x,d])*safelog(1-b[d,c])}))}) + safelog(probabilities[c])
        })}

# 4. FINDING Y HATS AND BUILDING CONFUSION MATRIX

# training data
train_data_scores <- fun1(x_train,pcd)
y_hat_train <- sapply(X=1:nrow(x_train), FUN = function(c) {which.max(train_data_scores[c,])})
train_confusion_matrix <- table(y_hat_train, y_train)

# test data
test_data_scores <- fun1(x_test,pcd)
y_hat_test <- sapply(X=1:nrow(x_test), FUN = function(c) {which.max(test_data_scores[c,])})
test_confusion_matrix <- table(y_hat_test, y_test)

# 5. PRINTING THE RESULTS

print(prior)
print(pcd[,1])
print(pcd[,2])
print(pcd[,3])
print(pcd[,4])
print(pcd[,5])
print(train_confusion_matrix)
print(test_confusion_matrix)
