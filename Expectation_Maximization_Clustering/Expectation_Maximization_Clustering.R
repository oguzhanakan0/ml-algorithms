library(MASS)
library(mixtools)
set.seed(541)
# mean parameters
class_means <- rbind( c(+2.5, +2.5),
                      c(-2.5, +2.5),
                      c(-2.5, -2.5),
                      c(+2.5, -2.5),
                      c(0, 0))
# covariance parameters
class_covariances <- array(c(+0.8, -0.6, -0.6, +0.8,
                             +0.8, +0.6, +0.6, +0.8,
                             +0.8, -0.6, -0.6, +0.8,
                             +0.8, +0.6, +0.6, +0.8,
                             +1.6, +0.0, +0.0, +1.6
                             ), c(2, 2, 5))
# sample sizes
class_sizes <- c(50, 50, 50, 50, 100)

# generate random samples
# points1 <- mvrnorm(n = class_sizes[1], mu = class_means[1,], Sigma = class_covariances[,,1])
# points2 <- mvrnorm(n = class_sizes[2], mu = class_means[2,], Sigma = class_covariances[,,2])
# points3 <- mvrnorm(n = class_sizes[3], mu = class_means[3,], Sigma = class_covariances[,,3])
# points4 <- mvrnorm(n = class_sizes[4], mu = class_means[4,], Sigma = class_covariances[,,4])
# points5 <- mvrnorm(n = class_sizes[5], mu = class_means[5,], Sigma = class_covariances[,,5])
# X <- rbind(points1, points2, points3, points4, points5)
# 
# X1 <- mvrnorm(50, c(+2.5, +2.5), matrix(c(.8, -.6, -.6, .8), 2, 2))
# X2 <- mvrnorm(50, c(-2.5, +2.5), matrix(c(.8, .6, .6, .8), 2, 2))
# X3 <- mvrnorm(50, c(-2.5, -2.5), matrix(c(.8, -.6, -.6, .8), 2, 2))
# X4 <- mvrnorm(50, c(+2.5, -2.5), matrix(c(.8, .6, .6, .8), 2, 2))
# X5 <- mvrnorm(100, c(0, 0), matrix(c(1.6, 0, 0, 1.6), 2, 2))
# X <- rbind(X1, X2, X3, X4,X5)

X1 <- mvrnorm(50, class_means[1,], Sigma = class_covariances[,,1])
X2 <- mvrnorm(50, class_means[2,], Sigma = class_covariances[,,2])
X3 <- mvrnorm(50, class_means[3,], Sigma = class_covariances[,,3])
X4 <- mvrnorm(50, class_means[4,], Sigma = class_covariances[,,4])
X5 <- mvrnorm(100, class_means[5,], Sigma = class_covariances[,,5])
X <- rbind(X1, X2, X3, X4,X5)


# k-means clustering
K = 5
D = 2
N = 300
# Step 0: function to obtain matrix B (just for convenience)
find_B <- function (cluster_values){
  B <- matrix(0, nrow = 300, ncol = K)
  for (i in 1:300){
    B[i,cluster_values[i]] <- 1
  }
  return(B)
}

# Step 1: assign means randomly 
means <- cbind(runif(5, min = -2.5, max = +2.5),runif(5, min = -2.5, max = +2.5))
distances <- sapply(1:5, function(c) {sqrt((X[,1]-means[c,1])^2+(X[,2]-means[c,2])^2)})
cluster_values <- sapply(1:300, function(i) { which.min(distances[i,])})
B <- find_B(cluster_values)

# Step 2: run k-means for 2 iterations
iter = 0
while(1){
  if(iter == 10){
    break
  }
  
  for (k in 1:K) {
    means[k,] <- colMeans(X[cluster_values == k,])
  }
  distances <- sapply(1:5, function(c) {sqrt((X[,1]-means[c,1])^2+(X[,2]-means[c,2])^2)})
  cluster_values <- sapply(1:300, function(i) { which.min(distances[i,])})
  B <- find_B(cluster_values)
  iter<- iter+1
}

# EM Algorithm
# Step 1. Initialize (find prior probabilities, class covariances, means)
prior <- sapply(1:5, function(c) {round(length(cluster_values[cluster_values==c])/length(cluster_values),3)})
Z <- B # to initiate success probabilities matrix, I used B matrix obtained from k-means
H <- B
class_covariances2 <- class_covariances
for (k in 1:5) {
  class_covariances2[,,k] <- rowSums(sapply(1:300, function(i) {H[i,k]*((X[i,]-means[k,])%*%t(X[i,]-means[k,]))}))/sum(H[,k])
}

while(1){
  if(iter==100){
    break
  }
  for(k in 1:K){
    for(i in 1:N){
      H[i,k]=dmvnorm(X[i,],means[k,],class_covariances2[,,k])*prior[k]/sum(sapply(1:K, function(c) {dmvnorm(X[i,],means[c,],class_covariances2[,,c])*prior[c]}))
    }
  }
  for(k in 1:K){
    means[k,] <- rowSums(sapply(1:300, function(i) {H[i,k]*X[i,]}))/sum(H[,k])
    class_covariances2[,,k] <- rowSums(sapply(1:300, function(i) {H[i,k]*((X[i,]-means[k,])%*%t(X[i,]-means[k,]))}))/sum(H[,k])
  }
  cluster_values <- sapply(1:N, function(c) {which.max(H[c,])})
  prior <- sapply(1:5, function(c) {sum(H[,c])/N})
  Z <- find_B(cluster_values)
  iter<-iter+1
}

# Step 3: plot the centers

colors <- c("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a")
margin = 2
plot(X[,1], X[,2], type = "p", pch = 19, col = colors[cluster_values], las = 1,
     xlab = "x1", ylab = "x2", main = "EM, iteration=100", 
     xlim = c(min(X[,1])-margin,max(X[,1])+margin),
     ylim = c(min(X[,2])-margin,max(X[,2])+margin))

for (c in 1:5){
  ellipse(mu=means[c,], sigma=class_covariances2[,,c], alpha = .05, npoints = 250, col="black",lwd = 2)
  ellipse(mu=class_means[c,], sigma=class_covariances[,,c], alpha = .05, npoints = 250, col="black", lwd = 2, lty = "dashed")
}