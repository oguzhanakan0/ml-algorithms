# Set wd and read data
data_set <- read.csv("hw05_data_set.csv")
# Get x and y values
x <- data_set$x
y <- data_set$y
x_train <- x[1:100]
y_train <- y[1:100]
x_test  <- x[101:133]
y_test  <- y[101:133]
remove(x,y)

tree <- function(P){
    
  # Parameters
  N_train <- length(y_train)
  N_test <- length(y_test)
  
  # create necessary data structures
  node_indices <- list()
  is_terminal <- c()
  need_split <- c()
  
  node_splits <- c()
  node_averages <- list()
  
  # put all training instances into the root node
  node_indices <- list(1:N_train)
  is_terminal <- c(FALSE)
  need_split <- c(TRUE)
  
  # learning algorithm
  while (1) {
    # find nodes that need splitting
    split_nodes <- which(need_split)
    
    if (length(split_nodes) == 0) {
      break
    }
    # iterate through splits
    
    for (split_node in split_nodes) {
      data_indices <- node_indices[[split_node]]
      need_split[split_node] <- FALSE
      node_averages[[split_node]] <- sum(y_train[data_indices])/length(data_indices)
      if(length(y_train[data_indices])<=P){
        is_terminal[split_node] <- TRUE
      } else {
        is_terminal[split_node] <- FALSE
        
        best_split <- c(0)
        unique_values <- sort(unique(x_train[data_indices]))
        split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2
        split_scores <- rep(0,length(split_positions))
        for (s in 1:length(split_positions)){
          left_indices <- data_indices[which(x_train[data_indices] < split_positions[s])]
          right_indices <- data_indices[which(x_train[data_indices] >= split_positions[s])]
          split_scores[s] <- (1/length(data_indices)*(sum((y_train[left_indices]-sum(y_train[left_indices])/length(left_indices))^2)+sum((y_train[right_indices]-sum(y_train[right_indices])/length(right_indices))^2)))
        }
        
      best_split <- split_positions[which.min(split_scores)]
      node_splits[split_node] <- best_split
      
      # create left node using the selected split
      left_indices <- data_indices[which(x_train[data_indices] < best_split)]
      node_indices[[2 * split_node]] <- left_indices
      is_terminal[2 * split_node] <- FALSE
      need_split[2 * split_node] <- TRUE
      
      # create right node using the selected split
      right_indices <- data_indices[which(x_train[data_indices] >= best_split)]
      node_indices[[2 * split_node + 1]] <- right_indices
      is_terminal[2 * split_node + 1] <- FALSE
      need_split[2 * split_node + 1] <- TRUE
      }
    }
  }
    
  # traverse tree for test data points
  y_predicted <- rep(0, N_test)
  for (i in 1:N_test) {
    index <- 1
    while (1) {
      if (is_terminal[index] == TRUE || is.na(node_splits[index])) {
        y_predicted[i] <- node_averages[[index]]
        break
      } 
      else {
        if (x_test[i] <= node_splits[index]) {
          index <- index * 2
        } else {
          index <- index * 2 + 1
        }
      }
    }
  }
  
  RMSE <- sqrt(sum((y_test-y_predicted)^2)/N_test)
  
  if(P==10){
    # extract rules
    terminal_nodes <- which(is_terminal)
    left_borders <- rep(0,length(terminal_nodes))
    right_borders <- rep(0,length(terminal_nodes))
    for (terminal_node in terminal_nodes) {
      i = 0
      index <- terminal_node
      rules_less_than <- c()
      rules_greater_than <- c()
      while (index > 1) {
        parent <- floor(index / 2)
        if (index %% 2 == 0) {
          # if node is left child of its parent
          rules_less_than <- c(node_splits[parent], rules_less_than)
        } else {
          # if node is right child of its parent
          rules_greater_than <- c(node_splits[parent], rules_greater_than)
        }
        index <- parent
      }
      a = tryCatch({
        left_borders[terminal_node] = max(rules_greater_than)},
        warning = function(warning_condition) {
        }, error = function(error_condition) {
        }, finally={
        })
      b = tryCatch({
        right_borders[terminal_node] = min(rules_less_than)},
        warning = function(warning_condition) {
        }, error = function(error_condition) {
        }, finally={
        })
      i=i+1
    }
    
    left_borders <- left_borders[which(is_terminal)]
    left_borders[which(!is.finite(left_borders))] <- 0
    
    right_borders <- right_borders[which(is_terminal)]
    right_borders[which(!is.finite(right_borders))] <- 60
    
    graph <- cbind(left_borders,right_borders,node_averages[which(is_terminal)])
    graph <- graph[order(left_borders),]
    left_borders <- graph[,1]
    right_borders <- graph[,2]
    y_values <- graph[,3]
    
    # Plot the data
    plot(x_train[1:100], y_train[1:100], type = "p", pch = 19, col = "blue", las = 1,
         xlab = "x", ylab = "y", main = "P = 10")
    points(x_test[1:33], y_test[1:33], type = "p", pch = 19, col = "red")
    for (b in 1:length(left_borders)) {
      lines(c(left_borders[[b]], right_borders[[b]]), c(y_values[[b]], y_values[[b]]), lwd = 2, col = "black")
      if (b < length(left_borders)) {
        lines(c(right_borders[[b]], right_borders[[b]]), c(y_values[[b]], y_values[[b+1]]), lwd = 2, col = "black") 
      }
    }
    
    #print RMSE
    cat("RMSE is", RMSE,"when P is",P)
  }
  return(RMSE)
}

rmses <- c()
for(i in 1:20){
  temp = tree(i)
  rmses = c(rmses,temp)
}

plot(1:length(rmses), rmses[1:length(rmses)],
     type = 'b', lwd = 2, las = 1, pch =1)
