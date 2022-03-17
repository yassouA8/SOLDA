
## generic functions
# 
beta.weights <- function(vari, y, classe){
  x1 <- vari[y==classe]
  x2 <- vari[y==0]
  x3 <- vari[(y!=0)&(y!=classe)]
  ks.test(x1, x3)$statistic  + ks.test(x2, x3)$statistic
}

#
change_levels <- function(x_factor, niveaux)
{
  Sorted.levels <- names(sort(table(x_factor), decreasing = T))
  x_factor <- relevel(x_factor, ref = Sorted.levels[1])
  levels(x_factor) <- niveaux
  x_factor
}

# 
orders <- function(xrow){
  order(xrow, decreasing = TRUE)[1]
} 


## main function
#' @title Sparse Overlapped Linear Discriminant Analysis (SOLDA)
#' @description A multi-class classifier using the Linear Discriminant Analysis (LDA) framework.
#'
#' @param y_data the response variable. \code{y_data} must be a factor.
#' @param x_data.train a numeric (train) data matrix with observations as rows and variables as columns. 
#' @param x_data.test a numeric (test) data matrix with observations as rows and variables as columns.
#' @param lambda1 tuning parameter for the L_1 penalty. If a single positive value is supplied, then it is considered as the optimal choice. If a sequence of positive values is supplied, then cross-validation is used to determine the optimal choice. If the "default" value is supplied, then the optimal choice is determined based on data supplied.
#' @param lambda2 tuning parameter for the L_2 penalty. To add later (default behavior for now)
#' @param weights adaptive weights numeric weights. To add later (default behavior for now)
#' @param num_folds number of folds to run cross-validation. Default is 10.
#' @param tolerance Default is 0.0000001.
#' @param max_iter Default is 100000.
#' @param num_threads number of cores to use for parallel computations. Default is 2.
#' @export
#' @examples
#'\dontrun{
#' ############### solda.fit function ###############
#' library(mvtnorm)
#' set.seed(1)
#' n1 <- 20
#' n2 <- 20
#' n3 <- 20
#' n <- n1 + n2 + n3
#' p <- 800
#' sigma <- diag(p)
#' mean1 <- c(rep(1,10), rep(0,p-10))
#' mean2 <- c(rep(0,10), rep(1,10), rep(0,p-20))
#' mean3 <- c(rep(0,20), rep(1,10), rep(0,p-30))
#' x_data.train <- rbind(rmvnorm(n1, mean1, sigma), rmvnorm(n2, mean2, sigma), rmvnorm(n3, mean3, sigma))
#' y_data <- as.factor(c(rep("1",n1), rep("2",n2), rep("3",n3)))
#' solda.output <- solda.fit(y_data = y_data, x_data.train = x_data.train, x_data.test = x_data.train)
#' solda.y <- solda.output$y_solda
#'}
solda.fit <- function(y_data, x_data.train, x_data.test, lambda1="default", lambda2=0.25, weights = "default",
                      num_folds = 10, tolerance = 0.0000001, max_iter = 100000, num_threads = 2){
  
  ## compute the nuber of classes in the data
  classes <- names(sort(table(y_data),decreasing = TRUE))
  Nclasses <- length(classes)
  y_data <- as.numeric(change_levels(y_data, (1:Nclasses)))-1
  
  ## compute the number of features
  p <- dim(x_data.train)[2]
  
  if(weights == "default"){
    ## compute the weigths as in solda manuscript 
    MV.matrix <- matrix(0,nrow = p, ncol = (Nclasses-1))
    for(kk in 1:(Nclasses-1)){
      MV.matrix[,kk] <- apply(x_data.train, 2, beta.weights, y=y_data, classe=kk)
    }  
    MV.matrix <- (2 - MV.matrix)*0.45 + 0.1
  }else{
    ## the user supplies the weights
    MV.matrix = weights
  }
  
  if(length(lambda1) > 1){
    ## here tha lambda sequence is determined by the user
    lambdas_diversity <- sort(lambda1, decreasing = TRUE)
    
    ## construct a grid with both lambda1 and lambda2
    lambdas_grid <- cbind(c(0*lambdas_diversity, 0.25*lambdas_diversity, 4*lambdas_diversity),
                          c(lambdas_diversity, lambdas_diversity, lambdas_diversity))
    
    ## run a cross validation and get cv-errors
    mse.Sparsity.Diversity <- CV_Ensemble_EN(X_data=x_data.train, y_data=y_data,
                                             lambdas_grid=lambdas_grid, Nclasses=Nclasses, num_folds=num_folds, tolerance=tolerance,
                                             max_iter=max_iter, num_threads=num_threads, MV_matrix=MV.matrix)
    
    ## pick the best lambda1 based on cv-erros (sparse solution is prefered)
    lamdas.opt.indices <- which(mse.Sparsity.Diversity==min(mse.Sparsity.Diversity), arr.ind = TRUE)
    lambdas_grid.opt <- matrix(lambdas_grid[lamdas.opt.indices[,1],], ncol=2)
    lamdas.opt <- lambdas_grid.opt[order(lambdas_grid.opt[,2], decreasing = TRUE)[1],]
    
  }else{ 
    if(lambda1 == "default"){
      ## here tha lambda1 sequence is determined from the data
      Means.matrix <- Meanscompute(x_data.train, y_data, Nclasses, p)
      Delta.matrix <- apply(Means.matrix[2:Nclasses,],1,function(x){sapply(x-Means.matrix[1,],abs)})
      max.point <- 1/min(MV.matrix)
      lambdas_diversity.opt <- max.point*max(Delta.matrix)
      
      ## construct a sequence of 100 lambda1 condidates 
      lambdas_diversity <- seq(lambdas_diversity.opt, 0.01*lambdas_diversity.opt, length.out=100)
      
      ## construct a grid with both lambda1 and lambda2
      lambdas_grid <- cbind(c(0*lambdas_diversity, 0.25*lambdas_diversity, 4*lambdas_diversity),
                            c(lambdas_diversity, lambdas_diversity, lambdas_diversity))
      
      ## run a cross validation and get cv-errors
      mse.Sparsity.Diversity <- CV_Ensemble_EN(X_data=x_data.train, y_data=y_data,
                                               lambdas_grid=lambdas_grid, Nclasses=Nclasses, num_folds=num_folds, tolerance=tolerance,
                                               max_iter=max_iter, num_threads=num_threads, MV_matrix=MV.matrix)
      
      ## pick the best lambda1 based on cv-erros (sparse solution is prefered)
      lamdas.opt.indices <- which(mse.Sparsity.Diversity==min(mse.Sparsity.Diversity), arr.ind = TRUE)
      lambdas_grid.opt <- matrix(lambdas_grid[lamdas.opt.indices[,1],], ncol=2)
      lamdas.opt <- lambdas_grid.opt[order(lambdas_grid.opt[,2], decreasing = TRUE)[1],]
    }else{
      ## the user supplies only one value for lambda1
      lamdas.opt <- lambda1
    }
  }
  ## estimate the matrix of bayes directions
  Theta.opt <- Ensemble_EN_Opt(x_data.train, Means.matrix, lamdas.opt[1], lamdas.opt[2],
                               (Nclasses-1), tolerance, max_iter, MV_matrix=MV.matrix)
  
  ## predict the label of the test dataset
  y_solda <- Predictions_lambdasolo(Means.matrix, y_data, x_data.test, Nclasses, Theta.opt)
  y_solda <- as.factor(as.numeric(y_solda))
  levels(y_solda) <- classes
  
  ##the output
  list(y_solda = y_solda, Theta.opt = Theta.opt)
} 

