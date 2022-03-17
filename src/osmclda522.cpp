#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]
#include "config.h"


// [[Rcpp::export]]
arma::mat Meanscompute(const arma::mat & X_data, const arma::vec & y_data, const arma::uword & Nclasses, const arma::uword & p)
{

  arma::mat Means_mat = zeros(Nclasses,p);
  for (uword classe = 0; classe < Nclasses; classe++) {
    Means_mat.row(classe) = mean(X_data.rows(find(y_data==classe)));
  }

  return Means_mat;
}

// Function to return the positive part of any number
double Positive_Part(const double & x){

  double positive = x;
  if (x <= 0){
    positive = 0;
  }
  return positive;
}

// Function to return the sign of any number - returns a sign of 0 if the numerical argument is 0
double Sign(const double & x){

  double sign = 0;
  if (x < 0){
    sign = -1;
  }
  if (x > 0){
    sign = 1;
  }
  return sign;
}

// Function that returns the absolute value of any number
double Absolute_Value(const double & x){

  double abs_value = x * Sign(x);
  return abs_value;
}

// Function that returns the numerical value from the soft-thresholding operator (takes 2 numerical values as input)
double Soft_Thresholding(const double & x,
                         const double & gamma){

  double soft = 0;
  soft = Sign(x) * Positive_Part(Absolute_Value(x) - gamma);
  return soft;
}


arma::vec beta_weights(const arma::mat & beta,
                       const arma::uword & group){
  // Computes weights for the l1 interaction penalty term
  // arma::uword num_groups = beta.n_cols;
  arma::vec sum_abs = zeros(beta.n_rows, 1);
  // arma::vec indices = ones(num_groups, 1);
  // indices[group] = 0;
  // sum_abs = beta * indices;
  sum_abs = beta.col(group);///sum(beta.col(group));
  return(sum_abs);
}


arma::uvec Set_Diff(const arma::uvec & big,
                    const arma::uvec & small){
  // Find set difference between a big and a small set of variables.
  // Note: small is a subset of big (both are sorted).
  int m = small.n_elem;
  int n = big.n_elem;
  arma::uvec test = uvec(n, fill::zeros);
  arma::uvec zeros = uvec(n - m, fill::zeros);

  for (int j = 0 ; j < m ; j++){
    test[small[j]] = small[j];
  }

  test = big - test;
  if(small[0] != 0){
    test[0] = 1;
  }
  zeros = find(test != 0);
  return(zeros);
}

void Cycling(const arma::mat & Sigma,
             const arma::vec & thresh,
             const arma::vec & stdz,
             const arma::uword & group,
             arma::mat & current_res,
             arma::mat & out_beta){

  // Does one cycle of CD for one group, updates the coefficient
  // in out_beta and the residuals in current_res
  // out_beta is a (p*G) matrix
  arma::uword p = Sigma.n_cols;
  double resid_corr = 0;
  double old_coef = 0;

  for(arma::uword j = 0; j < p; j++){
    old_coef = out_beta(j, group);
    out_beta(j, group) = 0;
    // Current residuals
    resid_corr = current_res(j,group) + Sigma(j,j)*old_coef;
    // Update
    out_beta(j, group) = Soft_Thresholding(resid_corr, thresh[j]) / stdz[j];
    if (out_beta(j, group) != old_coef){
      // Only update residuals if the coefficient changed
      current_res.col(group) += Sigma.col(j) * (old_coef - out_beta(j, group));
    }
  }
}


void Ensemble_EN_Solver(const arma::mat & Sigma,
                        const double & lambda_sparsity,
                        const double & lambda_diversity,
                        const arma::uword & num_groups,
                        const double & tolerance,
                        unsigned long & max_iter,
                        arma::mat & current_res,
                        arma::mat & beta,
						const arma::mat & MV_matrix){
  // Solves ensembles EN function for fixed penalty terms.
  // Input
  // Sigma: Covariance matrix of the data
  // lambdas_sparsity: penalty parameter for individual coefficients
  // lambdas_diversity: penalty parameter for interactions between groups
  // num_groups: number of groups
  // tolerance: tolerance parameter to stop the iterations
  // max_iter: maximum number of iterations before stopping the iterations over the groups
  //
  // # Output
  // beta: slopes
  // current_res: residuals
  arma::uword p = Sigma.n_cols;
  arma::vec thresh = zeros(p, 1);
  arma::vec stdz = zeros(p, 1);
  double conv_crit = 1;
  arma::uword iteration = 0;
  arma::mat beta_old = zeros(p, num_groups);

  beta_old = beta;
  stdz = Sigma.diag() + lambda_sparsity;
  // Do one cycle to start with
  iteration += 1;
  for (arma::uword group = 0; group < num_groups; group++){
    // Update penalty
    thresh = (lambda_diversity *beta_weights(MV_matrix, group));
    // Do one CD cycle
    Cycling(Sigma, thresh, stdz, group, current_res, beta);
  }
  beta_old = beta;
  // cout << '\n' << objective_new << '\n';
  while((conv_crit > tolerance) & (iteration <= max_iter)){
    iteration += 1;
     for (arma::uword group = 0; group < num_groups; group++){
       // Update penalty
       thresh = (lambda_diversity *beta_weights(MV_matrix, group));
       // Do one CD cycle
       Cycling(Sigma, thresh, stdz, group, current_res, beta);
    }
    conv_crit = square(mean(beta_old, 1) - mean(beta, 1)).max();
    // objective_old = objective_new;
    beta_old = beta;
  }
}



arma::cube Ensemble_EN_Grid(const arma::mat & X_data,
                            const arma::mat & Means_matrix,
                            const arma::mat & lambdas_grid,
                            const arma::uword & num_groups,
                            const double & tolerance,
                            unsigned long & max_iter,
							const arma::mat & MV_matrix){
  // Computes Ensemble EN over a path of penalty values
  //
  // Input
  // X_data design matrix
  // Means_matrix is a (Nclasses, p) matrix where row k contains the mean of class k
  // which_lambda: which penalty is the grid for? 1: lambda_sparsity, 2: lambda_diversity
  // lambdas_grid: grid of penalty values to compute the solution over
  // lambda_fixed: the other penalty
  // num_groups: number of groups
  // tolerance: tolerance parameter to stop the iterations
  // max_iter: maximum number of iterations before stopping
  //
  // Output
  // a cube whose slices are the slopes computer over lambda_grid

  arma::uword p = X_data.n_cols;
  arma::mat Sigma = zeros(p,p);
  Sigma = cov(X_data);
  // Delta is a (p*num_groups) matrix (muK - mu1)
  arma::mat Delta = zeros(p,num_groups);
  for (uword classe = 1; classe < (num_groups+1); classe++) {
    Delta.col(classe-1) = (Means_matrix.row(classe)-Means_matrix.row(0)).t();
  }

  arma::uword num_lambda = lambdas_grid.n_rows;
  // Slopes
  arma::mat beta_old_grid = zeros(p, num_groups);
  // Current model residuals for each group (by column), at each iteration and grid point
  arma::mat current_res = zeros(p, num_groups);
  // Output
  arma::cube out_betas = zeros(p, num_groups, num_lambda);
  // Residuals  for the empty model
  for (uword group = 0; group < num_groups; group++) {
    current_res.col(group) = Delta.col(group);
  }


  for (int i = (num_lambda - 1); i >= 0; i--){
    // Use the solver. Iterations start at beta_old_grid. Output
    // is written to beta_old_grid, residuals are updated in current_res
    Ensemble_EN_Solver(Sigma, lambdas_grid.at(i,0), lambdas_grid.at(i,1),
                      num_groups, tolerance, max_iter, current_res, beta_old_grid, MV_matrix);
    out_betas.slice(i) = beta_old_grid;
  }

  return(out_betas);
}

arma::uword Vec_mat_distance(const arma::mat & X_newi, const arma::mat & Means_mat, const arma::mat & Theta_lambda, const arma::uword & Nclasses)
{
  // function to copmpute the eucledienne distance between a vector and each row of a matrix
  // x_new_proj (1,Nclasses-1) is the projection of the new observation on THETA
  // Means_mat_proj is a matrix (Nclasses, Nclasses-1) that contains in row k the mean of a group k projected on THETA
  // Nclasses is the number of Nclasses
  //Output: we compute K_distances; a (Nclasses,1) vector where each elements is diff(xnew, mu_k), both of them projected on THETA
  // and return the index_min of it; the predicted label of xnew

  arma::vec K_distances = zeros(Nclasses,1);
  K_distances[0] = 0;
  for (uword classe = 1; classe < Nclasses; classe++) {
    K_distances[classe] = dot( (X_newi - 0.5*(Means_mat.row(classe)+Means_mat.row(0))).t() , Theta_lambda.col(classe-1) );
  }
  return index_max(K_distances);
}

// [[Rcpp::export]]
arma::vec Predictions_lambdasolo(const arma::mat & Means_mat, const arma::vec & y_data, const arma::mat & X_new, const arma::uword & Nclasses, const arma::mat & Theta_lambda)
{
  // function to compute predictions for fixed lambda
  // X_data is the matrix of observations (training)
  // y_ data is the vector of labels (training)
  // X_new is the matrix of new observations to which we want labels
  // Nclasses is tne number classes (not groups)
  // Theta_lambda is a (p,Num_groups) matrix contianing the estimated projections for a certain lambda
  // the output is a vector containing the prediction labels for the X_new observations
  uword Nnew = X_new.n_rows;
  //uword p = Means_mat.n_cols;
  arma::vec y_pred = zeros(Nnew, 1);
  //arma::mat Means_mat = Meanscompute(X_data, y_data,  Nclasses, p);
  for (uword i = 0; i < Nnew; i++) {
		y_pred[i] = Vec_mat_distance(X_new.row(i), Means_mat, Theta_lambda, Nclasses);
	}
  return y_pred;
}


arma::mat Predictions_lambdagrid(const arma::mat & Means_mat, const arma::vec & y_data, const arma::mat & X_new, const arma::uword & Nclasses, arma::cube & Theta)
{
  // function to compute predictions for a lambda.grid
  // X_data is the matrix of observations (training)
  // y_ data is the vector of labels (training)
  // X_new is the matrix of new observations to which we want labels
  // Nclasses is tne number classes (not groups)
  // Theta is a (p,Num_groups, numb_lambda.grid) cube contianing the estimated projections for a certain lambda in each slice
  // the output is a matrix where each column contains the prediction labels for the X_new observations with respect to a certain lambda in the lambda.grid


  arma::uword Nlamdas = Theta.n_slices;
  arma::uword Nnew = X_new.n_rows;
  arma::mat lamdas_predictions = zeros(Nnew, Nlamdas);
  for (uword nlambda = 0; nlambda < Nlamdas; nlambda++) {
      lamdas_predictions.col(nlambda) = Predictions_lambdasolo(Means_mat, y_data, X_new, Nclasses, Theta.slice(nlambda));
  }
  return lamdas_predictions;
}


// [[Rcpp::export]]
arma::vec CV_Ensemble_EN(const arma::mat & X_data,
                         const arma::vec & y_data,
                         const arma::mat & lambdas_grid,
                         const arma::uword & Nclasses,
                         const arma::uword & num_folds,
                         const double & tolerance,
                         unsigned long & max_iter,
                         const arma::uword & num_threads,
						 const arma::mat & MV_matrix){
  // Finds CV MSE for Ensemble EN with given penalty parameters
  // Input
  // x: design matrix, shuffled
  // y: responses, shuffled
  // which_lambda: which penalty is the grid for? 1: lambda_sparsity, 2: lambda_diversity
  // lambdas_grid: grid of penalty values to compute the solution over
  // lambda_fixed: the other penalty
  // num_groups: number of groups
  // num_folds: number of folds for CV
  // tolerance: tolerance parameter to stop the iterations
  // max_iter: maximum number of iterations before stopping
  // num_threads: number of threads for parallel computations
  //
  // Output
  // mse : the CV MSE for each lambda in lambda_grid
  const arma::uword num_groups = Nclasses -1;
  const arma::uword p = X_data.n_cols;
  const double n = X_data.n_rows;
  const arma::uword num_lambdas = lambdas_grid.n_rows;
  const arma::uvec indin = linspace<uvec>(0, n - 1, n);
  const arma::uvec inint = linspace<uvec>(0, n , num_folds + 1);
  arma::mat mses = zeros(num_lambdas, num_folds);
# pragma omp parallel for num_threads(num_threads)
  for(arma::uword fold = 0; fold < num_folds; fold++){
    // Get test and training samples
    arma::uvec test = linspace<uvec>(inint[fold], inint[fold + 1] - 1, inint[fold + 1] - inint[fold]);
    arma::uvec train = Set_Diff(indin, test);
    // Fit using train, predict using test

    arma::mat Means_matrix = Meanscompute(X_data.rows(train), y_data.rows(train),  Nclasses, p);

    arma::cube betas = zeros(p, num_groups, num_lambdas);
    betas = Ensemble_EN_Grid(X_data.rows(train), Means_matrix, lambdas_grid, num_groups, tolerance, max_iter, MV_matrix);

    arma::mat preds_ave = Predictions_lambdagrid(Means_matrix, y_data.rows(train), X_data.rows(test), Nclasses, betas);
    for(arma::uword i = 0; i < num_lambdas; i++){
      mses.at(i, fold) = accu( y_data.rows(test) != preds_ave.col(i) );
    }
  }
  arma::vec out = sum(mses, 1)/num_folds;
  return(out);
}

// [[Rcpp::export]]
arma::mat Ensemble_EN_Opt(const arma::mat & X_data,
                            const arma::mat & Means_matrix,
                            const double & lambdas_sparsity,
                            const double & lambda_diversity,
                            const arma::uword & num_groups,
                            const double & tolerance,
                            unsigned long & max_iter,
							const arma::mat & MV_matrix){
  // Gives theta for lambdas optimals
  //
  // Input
  // X_data design matrix
  // Means_matrix is a (Nclasses, p) matrix where row k contains the mean of class k
  // which_lambda: which penalty is the grid for? 1: lambda_sparsity, 2: lambda_diversity
  // lambdas_grid: grid of penalty values to compute the solution over
  // lambda_fixed: the other penalty
  // num_groups: number of groups
  // tolerance: tolerance parameter to stop the iterations
  // max_iter: maximum number of iterations before stopping
  //
  // Output
  // a cube whose slices are the slopes computed over lambda_grid

  arma::uword p = X_data.n_cols;
  arma::mat Sigma = zeros(p,p);
  Sigma = cov(X_data);
  // Delta is a (p*num_groups) matrix (muK - mu1)
  arma::mat Delta = zeros(p,num_groups);
  for (uword classe = 1; classe < (num_groups+1); classe++) {
    Delta.col(classe-1) = (Means_matrix.row(classe)-Means_matrix.row(0)).t();
  }

  // Slopes
  // Current model residuals for each group (by column), at each iteration and grid point
  arma::mat current_res = zeros(p, num_groups);
  // Output
  arma::mat out_betas = zeros(p, num_groups);
  // Residuals  for the empty model
  for (uword group = 0; group < num_groups; group++) {
    current_res.col(group) = Delta.col(group);
  }

  Ensemble_EN_Solver(Sigma, lambdas_sparsity, lambda_diversity,
                     num_groups, tolerance, max_iter, current_res, out_betas, MV_matrix);

  return(out_betas);
}
