#include <vector>
#include <Rcpp.h>
#include <cmath>

using namespace std;

#define THETA_EVAL 101
// THETA_EVAL must be odd.
#define TAU_EVAL 11
#define NU_EVAL 11

struct HAPT_params {
  int max_level; // number of levels in the tree
  int n_groups; // number of groups
  int tau_states, nu_states; // number of states of each
  double tau_lims[2]; // limits for tau, on the log scale
  double nu_lims[2]; // limits for nu, on the log scale
  vector< vector< double > > tau_transition; // transition matrix for markov dependency on tau
  vector< vector< double > > nu_transition; // the same for nu
};

void main_recursion_function(int remaining_levels, // how much deeper are we going to go
                             struct HAPT_params* p, // parameters of the HAPT
                             const vector<double>& limits, //lower and upper limits of this region
                             const vector<double>& data_values, // array of observed points
                             const vector<int>& group_labels, // array of group labels
                             const vector<bool>& included, // array of indicators that a given point is included in this region
                             vector< vector<double> >& partial_log_ML, // store the partial ML's for each combination of shrinkage states here
                               // outer tau, inner nu
                             vector< vector< vector<double> > >& mean_density, // store the mean density here
                               // tau, nu, sample space
                             vector< vector< vector< vector<double> > > >& sample_densities, // store the sample densities here
                               // tau, nu, group, sample space
                             vector< vector< vector<double> > >& variance_function_term1, // store the variance function here (term 1)
                             vector< vector< vector<double> > >& variance_function_term2 // store the variance function here (term 2)
                             )
{
  // basic setup
  double midpoint = (limits[0] + limits[1])/2;
  int grid_size = pow(2,p->max_level);
  int grid_start = int(grid_size*limits[0]);
  int grid_mid = int(grid_size*midpoint);
  int grid_end = int(grid_size*limits[1]);
  
  
  // count data points and set up child included vectors
  vector< int > niL(p->n_groups, 0);
  vector< int > niR(p->n_groups, 0);
  vector<bool> included_L(included.size(),false);
  vector<bool> included_R(included.size(),false);
  
  for(unsigned int i = 0; i < data_values.size(); i++) {
    if(included[i]) {
      if(data_values[i] <= midpoint) {
        niL[group_labels[i]-1]++;
        included_L[i] = true;
      } else {
        niR[group_labels[i]-1]++;
        included_R[i] = true;
      }
    }
  }
  
  if (remaining_levels > 0) {
    // recurse down another level first
    vector< vector< double > > partial_log_ML_L(p->tau_states, vector< double >(p->nu_states, 0));
    vector< vector< double > > partial_log_ML_R(p->tau_states, vector< double >(p->nu_states, 0));
    
    vector< vector< vector<double> > > mean_density_L;
    vector< vector< vector<double> > > mean_density_R;
    mean_density_L = mean_density_R = mean_density;
    
    vector< vector< vector< vector<double> > > > sample_densities_L;
    vector< vector< vector< vector<double> > > > sample_densities_R;
    sample_densities_L = sample_densities_R = sample_densities;
    
    vector< vector< vector<double> > > variance_function_term1_L;
    vector< vector< vector<double> > > variance_function_term1_R;
    variance_function_term1_L = variance_function_term1_R = variance_function_term1;
    
    vector< vector< vector<double> > > variance_function_term2_L;
    vector< vector< vector<double> > > variance_function_term2_R;
    variance_function_term2_L = variance_function_term2_R = variance_function_term2;
    
    vector< double > limits_L;
    vector< double > limits_R;
    limits_L = limits; limits_L[1] = midpoint;
    limits_R = limits; limits_R[0] = midpoint;
    
    main_recursion_function(remaining_levels - 1,
                            p,
                            limits_R,
                            data_values,
                            group_labels,
                            included_R,
                            partial_log_ML_R,
                            mean_density_R,
                            sample_densities_R,
                            variance_function_term1_R,
                            variance_function_term2_R
    );
    main_recursion_function(remaining_levels - 1,
                            p,
                            limits_L,
                            data_values,
                            group_labels,
                            included_L,
                            partial_log_ML_L,
                            mean_density_L,
                            sample_densities_L,
                            variance_function_term1_L,
                            variance_function_term2_L
    );

    
    // pass up results from children
    
    for(int i = 0; i < p->tau_states; i++) {
      for(int j = 0; j < p->nu_states; j++) {
        // calculate posterior transition probs.
        vector< vector< double > > posterior_transition_probability_L(p->tau_states, vector< double >(p->nu_states, 0));
        vector< vector< double > > posterior_transition_probability_R(p->tau_states, vector< double >(p->nu_states, 0));
        double max_child_partial_log_ML_L =  partial_log_ML_L[i][j];
        double max_child_partial_log_ML_R =  partial_log_ML_R[i][j];
        for(int k = i; k < p->tau_states; k++) {
          for(int l = j; l < p->nu_states; l++) {
            if(partial_log_ML_L[k][l] > max_child_partial_log_ML_L)
              max_child_partial_log_ML_L = partial_log_ML_L[k][l];
            if(partial_log_ML_R[k][l] > max_child_partial_log_ML_R)
              max_child_partial_log_ML_R = partial_log_ML_R[k][l];
          }
        }
        double children_ML_L = 0;
        double children_ML_R = 0;
        //Rcpp::Rcout << "C_tau = " << i << ", C_nu = " << j << ": ";
        for(int k = i; k < p->tau_states; k++) {
          for(int l = j; l < p->nu_states; l++) {
            posterior_transition_probability_L[k][l] = p->tau_transition[i][k]*p->nu_transition[j][l] * exp(partial_log_ML_L[k][l] - max_child_partial_log_ML_L);
            posterior_transition_probability_R[k][l] = p->tau_transition[i][k]*p->nu_transition[j][l] * exp(partial_log_ML_R[k][l] - max_child_partial_log_ML_R);
            children_ML_L += posterior_transition_probability_L[k][l];
            children_ML_R += posterior_transition_probability_R[k][l];
            //Rcpp::Rcout << partial_log_ML_L[k][l] << ", ";
          }
        }
        //Rcpp::Rcout << endl;
        //Rcpp::Rcout << children_ML_L << ", " << children_ML_R << endl;

        //Rcpp::Rcout << "Posterior transition: ";
        for(int k = i; k < p->tau_states; k++) {
          for(int l = j; l < p->nu_states; l++) {
            posterior_transition_probability_L[k][l] /= children_ML_L;
            posterior_transition_probability_R[k][l] /= children_ML_R;
            
            //Rcpp::Rcout << posterior_transition_probability_L[k][l] << ", ";
          }
        }
        //Rcpp::Rcout << endl;
        
        
        // actually do the calculations now

        vector< double > children_mean_density_L(grid_size, 0);
        vector< double > children_mean_density_R(grid_size, 0);
        vector< vector< double > > children_sample_densities_L(p->n_groups, vector< double >(grid_size, 0));
        vector< vector< double > > children_sample_densities_R(p->n_groups, vector< double >(grid_size, 0));
        vector< double > children_variance_function_term1_L(grid_size, 0);
        vector< double > children_variance_function_term1_R(grid_size, 0);
        vector< double > children_variance_function_term2_L(grid_size, 0);
        vector< double > children_variance_function_term2_R(grid_size, 0);
        for(int k = 0; k < p->tau_states; k++) {
          for(int l = 0; l < p->nu_states; l++) {
            for(int m = 0; m < grid_size; m++) {
              children_mean_density_L[m] += posterior_transition_probability_L[k][l] * //
                mean_density_L[k][l][m];
              children_mean_density_R[m] += posterior_transition_probability_R[k][l] * //
                mean_density_R[k][l][m];
              for(int n = 0; n < p->n_groups; n++) {
                children_sample_densities_L[n][m] += posterior_transition_probability_L[k][l] * //
                  sample_densities_L[k][l][n][m];
                children_sample_densities_R[n][m] += posterior_transition_probability_R[k][l] * //
                  sample_densities_R[k][l][n][m];
              }
              children_variance_function_term1_L[m] += posterior_transition_probability_L[k][l] * //
                variance_function_term1_L[k][l][m];
              children_variance_function_term1_R[m] += posterior_transition_probability_R[k][l] * //
                variance_function_term1_R[k][l][m];
              children_variance_function_term2_L[m] += posterior_transition_probability_L[k][l] * //
                variance_function_term2_L[k][l][m];
              children_variance_function_term2_R[m] += posterior_transition_probability_R[k][l] * //
                variance_function_term2_R[k][l][m];
            }
          }
        }
        partial_log_ML[i][j] += log(children_ML_L) + log(children_ML_R) + //
          max_child_partial_log_ML_L + max_child_partial_log_ML_R;
        for(int m = 0; m < grid_size; m++) {
          mean_density[i][j][m] *= children_mean_density_L[m] * children_mean_density_R[m];
          for(int n = 0; n < p->n_groups; n++) {
            sample_densities[i][j][n][m] *= children_sample_densities_L[n][m] * children_sample_densities_R[n][m];
          }
          variance_function_term1[i][j][m] *= children_variance_function_term1_L[m] * children_variance_function_term1_R[m];
          variance_function_term2[i][j][m] *= children_variance_function_term2_L[m] * children_variance_function_term2_R[m];
        }
      }
    }
  }
  
  // do quadrature
  vector< vector<double> > theta_tau_value(THETA_EVAL, vector<double>((p->tau_states-1) * (TAU_EVAL - 1) + 1, 0));
  vector< vector<double> > theta_nu_value(THETA_EVAL, vector<double>((p->nu_states-1) * (NU_EVAL - 1) + 1, 0));
  vector< double > theta_nu_special_value((p->nu_states-1) * (NU_EVAL - 1) + 1, 0);
  
  vector< double > theta_eval_points(THETA_EVAL, 0);
  vector< double > tau_eval_points((p->tau_states-1) * (TAU_EVAL - 1) + 1, 0);
  vector< double > nu_eval_points((p->nu_states-1) * (NU_EVAL - 1) + 1, 0);
  
  // initialize eval points
  //Rcpp::Rcout << "theta eval points: ";
  for(unsigned int i = 0; i < theta_eval_points.size(); i++) {
    theta_eval_points[i] = 0.0001 + double(i)/(theta_eval_points.size()-1)*0.9998;
    //Rcpp::Rcout << theta_eval_points[i] << ", ";
  }
  //Rcpp::Rcout << endl;
  // tau
  for(int i = 0; i < (p->tau_states-1); i++) {
    tau_eval_points[i*(TAU_EVAL-1)] = pow(10,p->tau_lims[0] + double(i)/(p->tau_states-1)*(p->tau_lims[1] - p->tau_lims[0]));
    for(int j = 1; j < TAU_EVAL - 1; j++) {
      tau_eval_points[i*(TAU_EVAL-1)+j] = tau_eval_points[i*(TAU_EVAL-1)] + double(j)/(TAU_EVAL - 1)*(pow(10,p->tau_lims[0] + double(i+1)/(p->tau_states-1)*(p->tau_lims[1] - p->tau_lims[0])) - pow(10,p->tau_lims[0] + double(i)/(p->tau_states-1)*(p->tau_lims[1] - p->tau_lims[0])));
    }
  }
  tau_eval_points[tau_eval_points.size()-1] = pow(10,p->tau_lims[1]); 
  // nu
  for(int i = 0; i < (p->nu_states-1); i++) {
    nu_eval_points[i*(NU_EVAL-1)] = pow(10,p->nu_lims[0] + double(i)/(p->nu_states-1)*(p->nu_lims[1] - p->nu_lims[0]));
    for(int j = 1; j < NU_EVAL - 1; j++) {
      nu_eval_points[i*(NU_EVAL-1)+j] = nu_eval_points[i*(NU_EVAL-1)] + double(j)/(NU_EVAL - 1)*(pow(10,p->nu_lims[0] + double(i+1)/(p->nu_states-1)*(p->nu_lims[1] - p->nu_lims[0])) - pow(10,p->nu_lims[0] + double(i)/(p->nu_states-1)*(p->nu_lims[1] - p->nu_lims[0])));
    }
  }
  nu_eval_points[nu_eval_points.size()-1] = pow(10,p->nu_lims[1]); 
  
  /*
  Rcpp::Rcout << "nu eval points: ";
  for(int i = 0; i < nu_eval_points.size(); i++) {
    Rcpp::Rcout << nu_eval_points[i] << " ";
  }
  Rcpp::Rcout << endl;
   */
  
  // evaluate the factorized unnormalized posterior on the union of two planes
  for(unsigned int i = 0; i < theta_eval_points.size(); i++) {
    double theta = theta_eval_points[i];
    for(unsigned int j = 0; j < tau_eval_points.size(); j++) {
      double tau = tau_eval_points[j];
      theta_tau_value[i][j] = p->n_groups*(lgamma(tau) - lgamma(theta*tau) - lgamma((1-theta)*tau)) - tau - (p->tau_lims[1] - p->tau_lims[0])/(p->tau_states-1);
      for(unsigned int k = 0; k < niL.size(); k++) {
        theta_tau_value[i][j] += lgamma(theta*tau + niL[k]) + lgamma((1-theta)*tau + niR[k]) - lgamma(tau + niL[k] + niR[k]);
      }
    }
    // Rcpp::Rcout << "theta nu values: ";
    for(unsigned int j = 0; j < nu_eval_points.size(); j++) {
      double nu = nu_eval_points[j];
      theta_nu_value[i][j] = lgamma(nu) - 2*lgamma(nu/2) + log(theta) * (0.5 * nu - 1) + log(1-theta)*(0.5*nu - 1) - nu - (p->nu_lims[1] - p->nu_lims[0])/(p->nu_states-1);
      // Rcpp::Rcout << theta_nu_value[i][j] << ", ";

    }
    // Rcpp::Rcout << endl;
  }
  // Rcpp::Rcout << endl;
  for(unsigned int j = 0; j < nu_eval_points.size(); j++) {
    double nu = nu_eval_points[j];
    theta_nu_special_value[j] = lgamma(nu) - 2*lgamma(nu/2);
    int nL = 0; int nR = 0;
    for(int k = 0; k < p->n_groups; k++) {
      nL += niL[k]; nR += niR[k];
    }
    theta_nu_special_value[j] += lgamma(nu/2 + nL) + lgamma(nu/2 + nR) - lgamma(nu + nL + nR) - nu - (p->nu_lims[1] - p->nu_lims[0])/(p->nu_states-1);
  }
  
  // For each state, evaluate the integral
  vector< double > theta_weights(THETA_EVAL, 1.0/(THETA_EVAL - 1));
  vector< double > tau_weights(TAU_EVAL, 1.0/(TAU_EVAL - 1));
  vector< double > nu_weights(NU_EVAL, 1.0/(NU_EVAL - 1));
  theta_weights[0] /= 2;
  theta_weights[THETA_EVAL-1] /= 2;
  tau_weights[0] /= 2;
  tau_weights[TAU_EVAL-1] /= 2;
  nu_weights[0] /= 2;
  nu_weights[NU_EVAL-1] /= 2;
  
  
  for(int i = 0; i < p->tau_states - 1; i++) {
    for(int j = 0; j < p->nu_states - 1; j++) {
      int tau_offset = i*(TAU_EVAL-1);
      int nu_offset  = j*(NU_EVAL-1);
      double tau_interval_width = tau_eval_points[tau_offset + TAU_EVAL - 1] - tau_eval_points[tau_offset];
      double nu_interval_width = nu_eval_points[nu_offset + NU_EVAL - 1] - nu_eval_points[nu_offset];
      
      // Find normalizing values for numerical stability
      double max_theta_tau, max_theta_nu;
      
      max_theta_tau = theta_tau_value[0][tau_offset];
      max_theta_nu = theta_nu_value[0][nu_offset];
      for(int k = 0; k < THETA_EVAL; k++) {
        // find max_theta_tau, max_theta_nu
        for(int l = 0; l < TAU_EVAL; l++) {
          if(theta_tau_value[k][tau_offset + l] > max_theta_tau)
            max_theta_tau = theta_tau_value[k][tau_offset + l];
        }
        for(int l = 0; l < NU_EVAL; l++) {
          if(theta_nu_value[k][nu_offset + l] > max_theta_nu)
            max_theta_nu = theta_nu_value[k][nu_offset + l];
        }
      }

      //Rcpp::Rcout << "max_theta_tau: " << max_theta_tau << " max_theta_nu: " << max_theta_nu << endl;
      
      double theta_E1_value = 0;
      double theta_Etheta_value = 0;
      double theta_Etheta2_value = 0;
      double theta_Ecomptheta2_value = 0;
      double theta_Ethetatau_value = 0;
      double theta_Ecompthetatau_value = 0;
      double theta_Ethetatimestau_value = 0;
      double theta_Etau_value = 0;

      for(int k = 0; k < THETA_EVAL; k++) { 
        double theta = theta_eval_points[k];
        double tau_E1_value = 0;
        double nu_E1_value = 0;
        double tau_Ethetatau_value = 0;
        double tau_Ecompthetatau_value = 0;
        double tau_Ethetatimestau_value = 0;
        double tau_Etau_value = 0;
        // Rcpp::Rcout << "tau_E1_value: ";
        for(int l = 0; l < TAU_EVAL; l++) {
          double tau = tau_eval_points[tau_offset + l];
          tau_E1_value += tau_weights[l] * tau_interval_width * exp(theta_tau_value[k][tau_offset + l] - max_theta_tau);
          tau_Ethetatau_value += tau_weights[l] * tau_interval_width * exp(theta_tau_value[k][tau_offset + l] - max_theta_tau) * theta * (theta * tau + 1) / (tau + 1);
          tau_Ecompthetatau_value += tau_weights[l] * tau_interval_width * exp(theta_tau_value[k][tau_offset + l] - max_theta_tau) * (1-theta) * ((1-theta) * tau + 1) / (tau + 1);
          tau_Ethetatimestau_value += tau_weights[l] * tau_interval_width * exp(theta_tau_value[k][tau_offset + l] - max_theta_tau) * theta * tau;
          tau_Etau_value += tau_weights[l] * tau_interval_width * exp(theta_tau_value[k][tau_offset + l] - max_theta_tau) * tau;
          // Rcpp::Rcout << tau_E1_value << " ";
          if(l==0) {
            // Rcpp::Rcout << "weight: " << tau_weights[l] << "; theta_tau_value: " << theta_tau_value[k][tau_offset + l] << endl;
          }
        }
        // Rcpp::Rcout << endl;
         
        // Rcpp::Rcout << "nu_E1_value: ";
        for(int l = 0; l < NU_EVAL; l++) {
          nu_E1_value += nu_weights[l] * nu_interval_width * exp(theta_nu_value[k][nu_offset + l] - max_theta_nu);
          // Rcpp::Rcout << nu_E1_value << " ";
        }
        // Rcpp::Rcout << endl << endl;
        
        theta_E1_value += theta_weights[k] * tau_E1_value * nu_E1_value;
        theta_Etheta_value += theta_weights[k] * tau_E1_value * nu_E1_value * theta;
        theta_Etheta2_value += theta_weights[k] * tau_E1_value * nu_E1_value * theta * theta;
        theta_Ecomptheta2_value += theta_weights[k] * tau_E1_value * nu_E1_value * (1-theta) * (1-theta);
        theta_Ethetatau_value += theta_weights[k] * tau_Ethetatau_value * nu_E1_value;
        theta_Ecompthetatau_value += theta_weights[k] * tau_Ecompthetatau_value * nu_E1_value;
        theta_Ethetatimestau_value += theta_weights[k] * tau_Ethetatimestau_value * nu_E1_value;
        theta_Etau_value += theta_weights[k] * tau_Etau_value * nu_E1_value;
      }
      partial_log_ML[i][j] += log(theta_E1_value) + max_theta_nu + max_theta_tau;
      //Rcpp::Rcout << "partial_log_ML: " << partial_log_ML[i][j] << endl;
      
      // process theta_Etheta_value to obtain mean function
      double Etheta = theta_Etheta_value/theta_E1_value;
      double Ethetatimestau = theta_Ethetatimestau_value/theta_E1_value;
      double Etau = theta_Etau_value/theta_E1_value;
      
      /*if(limits[1] < .01) {
        Rcpp::Rcout << "Etheta: " << Etheta << "; C_nu" << j << endl;
        for(int k = 0; k < THETA_EVAL; k++) {
          for(int l = 0; l < NU_EVAL; l++) {
            //Rcpp::Rcout << theta_nu_value[k][l] << " ";
            Rcpp::Rcout << nu_weights[l] * nu_interval_width * exp(theta_nu_value[k][nu_offset + l] - max_theta_nu) << " ";
          }
          Rcpp::Rcout << endl;
        }
      }*/
        
      
      // it would be better to do this with slices using iterators
      for(int k = grid_start; k < grid_mid; k++) {
        mean_density[i][j][k] *= Etheta*2;
        for(int l = 0; l < p->n_groups; l++) {
          sample_densities[i][j][l][k] *= 2* (Ethetatimestau + niL[l])/(Etau + niL[l] + niR[l]);
        }
      }
      for(int k = grid_mid; k < grid_end; k++) {
        mean_density[i][j][k] *= (1-Etheta)*2;
        for(int l = 0; l < p->n_groups; l++) {
          sample_densities[i][j][l][k] *= 2 * ((Etau - Ethetatimestau) + niR[l])/(Etau + niL[l] + niR[l]);
        }
      }
      
      // process variance function
      double Ethetatau = theta_Ethetatau_value/theta_E1_value;
      double Ecompthetatau = theta_Ecompthetatau_value/theta_E1_value;
      double Etheta2 = theta_Etheta2_value/theta_E1_value;
      double Ecomptheta2 = theta_Ecomptheta2_value/theta_E1_value;
      
      for(int k = grid_start; k < grid_mid; k++) {
        variance_function_term1[i][j][k] *= Ethetatau * 4;
        variance_function_term2[i][j][k] *= Etheta2 * 4;
      }
      for(int k = grid_mid; k < grid_end; k++) {
        variance_function_term1[i][j][k] *= Ecompthetatau * 4;
        variance_function_term2[i][j][k] *= Ecomptheta2 * 4;
      }
    }
  }
  
  // Handle complete shrinkage states
  double tau_E1_special_value = 0;
  double tau_Etau_special_value = 0;
  double tau_Etauvar_special_value = 0;
  double nu_E1_special_value = 0;
  double nu_Etheta_special_value = 0;
  double nu_Etheta2_special_value = 0;
  double nu_Ecomptheta2_special_value = 0;
  
  // Case 1: tau = infinity, nu < infinity      
  
  for(int j = 0; j < p->nu_states - 1; j++) {
    int nu_offset  = j*(NU_EVAL-1);
    double nu_interval_width = nu_eval_points[nu_offset + NU_EVAL - 1] - nu_eval_points[nu_offset];
    double max_theta_nu_special;  
    
    // find maximum for normalization
    max_theta_nu_special = theta_nu_special_value[nu_offset];
    for(int l = 0; l < NU_EVAL; l++) {
      if(theta_nu_special_value[nu_offset + l] > max_theta_nu_special)
        max_theta_nu_special = theta_nu_special_value[nu_offset + l];
    }

    // integration
    int nL = 0; int nR = 0;
    for(int k = 0; k < p->n_groups; k++) {
      nL += niL[k]; nR += niR[k];
    }
    
    for(int l = 0; l < NU_EVAL; l++) {
      double nu = nu_eval_points[nu_offset + l];
      nu_E1_special_value += nu_weights[l] * exp(theta_nu_special_value[nu_offset + l] - max_theta_nu_special) * nu_interval_width;

      double nu_Etheta = (nu/2 + nL) / (nu + nL + nR);
      nu_Etheta_special_value += nu_weights[l] * exp(theta_nu_special_value[nu_offset + l] - max_theta_nu_special) * nu_interval_width * nu_Etheta;
      nu_Etheta2_special_value += nu_weights[l] * exp(theta_nu_special_value[nu_offset + l] - max_theta_nu_special) * nu_interval_width * nu_Etheta * nu_Etheta;
      nu_Ecomptheta2_special_value += nu_weights[l] * exp(theta_nu_special_value[nu_offset + l] - max_theta_nu_special) * nu_interval_width * (1 - nu_Etheta) * (1 - nu_Etheta);
    }
    partial_log_ML[p->tau_states-1][j] += log(nu_E1_special_value) + max_theta_nu_special;
    double Etheta = nu_Etheta_special_value / nu_E1_special_value;
    double Etheta2 = nu_Etheta2_special_value / nu_E1_special_value;
    double Ecomptheta2 = nu_Ecomptheta2_special_value / nu_E1_special_value;
    
    
    // process mean functions
    for(int k = grid_start; k < grid_mid; k++) {
      mean_density[p->tau_states-1][j][k] *= Etheta*2;
      for(int l = 0; l < p->n_groups; l++) {
        sample_densities[p->tau_states-1][j][l][k] = mean_density[p->tau_states-1][j][k];
      }
    }
    for(int k = grid_mid; k < grid_end; k++) {
      mean_density[p->tau_states-1][j][k] *= (1-Etheta)*2;
      for(int l = 0; l < p->n_groups; l++) {
        sample_densities[p->tau_states-1][j][l][k] = mean_density[p->tau_states-1][j][k];
      }
    }
    
    // process variance functions
    for(int k = grid_start; k < grid_mid; k++) {
      variance_function_term1[p->tau_states-1][j][k] *= 4 * Etheta2;
      variance_function_term2[p->tau_states-1][j][k] *= 4 * Etheta2;
    }
    for(int k = grid_mid; k < grid_end; k++) {
      variance_function_term1[p->tau_states-1][j][k] *= 4 * Ecomptheta2;
      variance_function_term2[p->tau_states-1][j][k] *= 4 * Ecomptheta2;
    }
  }

  // Case 2: tau < infinity, nu = infinity
  
  for(int i = 0; i < p->tau_states - 1; i++) {
    int tau_offset  = i*(NU_EVAL-1);
    double tau_interval_width = tau_eval_points[tau_offset + TAU_EVAL - 1] - tau_eval_points[tau_offset];
    double max_half_tau;
    
    // find maximum for normalization
    max_half_tau = theta_tau_value[(THETA_EVAL + 1)/2][tau_offset];
    for(int l = 0; l < TAU_EVAL; l++) {
      if(theta_tau_value[(THETA_EVAL + 1)/2][tau_offset + l] > max_half_tau)
        max_half_tau = theta_tau_value[(THETA_EVAL + 1)/2][tau_offset + l];
    }
    
    // integration
    
    for(int l = 0; l < TAU_EVAL; l++) {
      double tau = tau_eval_points[tau_offset + l];
      tau_E1_special_value += tau_weights[l] * tau_interval_width * exp(theta_tau_value[(THETA_EVAL + 1)/2][tau_offset + l] - max_half_tau);
      tau_Etau_special_value += tau_weights[l] * tau_interval_width * exp(theta_tau_value[(THETA_EVAL + 1)/2][tau_offset + l] - max_half_tau) * tau;
      tau_Etauvar_special_value += tau_weights[l] * tau_interval_width * exp(theta_tau_value[(THETA_EVAL + 1)/2][tau_offset + l] - max_half_tau) * (1 + 1.0/(tau + 1));
      }
    partial_log_ML[i][p->nu_states-1] += log(tau_E1_special_value) + max_half_tau;
    double Etau = tau_Etau_special_value / tau_E1_special_value;
    double Etauvar = tau_Etauvar_special_value / tau_E1_special_value;
    
    // process mean functions
    for(int k = grid_start; k < grid_mid; k++) {
      for(int l = 0; l < p->n_groups; l++) {
        sample_densities[i][p->nu_states-1][l][k] *= 2* (Etau/2 + niL[l])/(Etau + niL[l] + niR[l]);
      }
    }
    for(int k = grid_mid; k < grid_end; k++) {
      for(int l = 0; l < p->n_groups; l++) {
        sample_densities[i][p->nu_states-1][l][k] *= 2* (Etau/2 + niR[l])/(Etau + niL[l] + niR[l]);
      }
    }
    
    // process variance functions
    for(int k = grid_start; k < grid_mid; k++) {
      variance_function_term1[i][p->nu_states-1][k] *= Etauvar;
      // second term does not need to be processed
    }
    for(int k = grid_mid; k < grid_end; k++) {
      variance_function_term1[i][p->nu_states-1][k] *= Etauvar;
    }
  }
    
  // Case 3: tau = nu = infinity

  int nL = 0; int nR = 0;
  for(int k = 0; k < p->n_groups; k++) {
    nL += niL[k]; nR += niR[k];
  }
  partial_log_ML[p->tau_states-1][p->nu_states-1] += (nL + nR)*log(.5);
  
  // mean and variance functions are untouched.
  
  /*
  Rcpp::Rcout << "Partial log ML: " << endl;
  for(int i = 0; i < p->tau_states; i++) {
    for(int j = 0; j < p->nu_states; j++) {
      Rcpp::Rcout << partial_log_ML[i][j] << " ";
    }
    Rcpp::Rcout << endl;
  }
  Rcpp::Rcout << endl;
  */
}



// [[Rcpp::export]]
Rcpp::List HAPT(Rcpp::NumericVector x, Rcpp::IntegerVector groups, 
                    int maxlevel = 10, int nu_states = 4, int tau_states = 4, 
                    Rcpp::NumericVector nu_lims = Rcpp::NumericVector::create(0.0,4.0), 
                    Rcpp::NumericVector tau_lims = Rcpp::NumericVector::create(0.0,4.0),
                    double beta_nu = 1, double beta_tau = 1 ) {
  
  vector< double > C_x;
  vector< int > C_groups;
  for(int i = 0; i < x.size(); i++) {
    C_x.push_back(double(x[i]));
    C_groups.push_back(int(groups[i]));
  }
  
  // count how many groups we have
  int g = 0;
  for(unsigned int i = 0; i < C_groups.size(); i++) {
    if(C_groups[i] > g) 
      g = C_groups[i];
  }
  
  HAPT_params p;
  p.max_level = maxlevel;
  p.n_groups = g;
  p.nu_lims[0] = double(nu_lims[0]); p.nu_lims[1] = double(nu_lims[1]);
  p.tau_lims[0] = double(tau_lims[0]); p.tau_lims[1] = double(tau_lims[1]);
  p.nu_states = nu_states; p.tau_states = tau_states;

  // generate transition matrices
  for(int i = 0; i < p.nu_states; i++) {
    vector<double> nu_transition_row(p.nu_states, 0);
    double row_sum = 0;
    for(int j = i; j < p.nu_states; j++) {
      nu_transition_row[j] = exp(beta_nu * (i - j));
      row_sum += nu_transition_row[j];
    }
    for(int j = i; j < p.nu_states; j++) {
      nu_transition_row[j] /= row_sum;
    }
    p.nu_transition.push_back(nu_transition_row);
  }
  for(int i = 0; i < p.tau_states; i++) {
    vector<double> tau_transition_row(p.tau_states, 0);
    double row_sum = 0;
    for(int j = i; j < p.tau_states; j++) {
      tau_transition_row[j] = exp(beta_tau * (i - j));
      row_sum += tau_transition_row[j];
    }
    for(int j = i; j < p.tau_states; j++) {
      tau_transition_row[j] /= row_sum;
    }
    p.tau_transition.push_back(tau_transition_row);
  }

  
  vector<bool> included(x.size(), true);
  vector< vector<double> > log_ML(p.tau_states, vector< double >(p.nu_states, 0));
  vector< vector< vector<double> > > mean_density(p.tau_states, vector< vector< double > >(p.nu_states, vector< double >(pow(2,maxlevel), 1)));
  vector< vector< vector< vector<double> > > > sample_densities(p.tau_states, vector< vector< vector < double > > >(p.nu_states, vector< vector < double > >(g, vector< double >(pow(2,maxlevel), 1))));
  vector< vector< vector<double> > > variance_function_term1(p.tau_states, vector< vector< double > >(p.nu_states, vector< double >(pow(2,maxlevel), 1)));
  vector< vector< vector<double> > > variance_function_term2(p.tau_states, vector< vector< double > >(p.nu_states, vector< double >(pow(2,maxlevel), 1)));
  
  vector< double > limits;
  limits.push_back(0); limits.push_back(1);
  
  main_recursion_function(maxlevel - 1, 
                          &p,
                          limits,
                          C_x,
                          C_groups,
                          included,
                          log_ML,
                          mean_density,
                          sample_densities,
                          variance_function_term1,
                          variance_function_term2
                          );
  
  double return_log_ML = 0;
  vector< double > return_mean_density(pow(2,maxlevel), 0);
  vector< vector< double > > return_sample_densities(g, vector< double >(pow(2,maxlevel), 0));
  vector< double > return_variance_function(pow(2,maxlevel), 0);
  
  // normalize for numerical stability
  double max_log_ML = log_ML[0][0];
  for(int i = 0; i < p.tau_states; i++) {
    for(int j = 0; j < p.nu_states; j++) {
      if(log_ML[i][j] > max_log_ML)
        max_log_ML = log_ML[i][j];
    }
  }
  
  // calculate the overall marginal likelihood
  for(int i = 0; i < p.tau_states; i++) {
    for(int j = 0; j < p.nu_states; j++) {
      return_log_ML += p.tau_transition[0][i] * p.nu_transition[0][j] * exp(log_ML[i][j] - max_log_ML);
    }
  }
  return_log_ML = log(return_log_ML) + max_log_ML;
  
  // calculate posterior probabilities of states
  vector< vector< double > > posterior_state_prob(p.tau_states, vector< double >(p.nu_states, 0));
  double state_prob_sum = 0;
  // Rcpp::Rcout << "Posterior state probs: ";
  for(int i = 0; i < p.tau_states; i++) {
    for(int j = 0; j < p.nu_states; j++) {
      posterior_state_prob[i][j] = p.tau_transition[0][i] * p.nu_transition[0][j] * exp(log_ML[i][j] - max_log_ML);
      state_prob_sum += posterior_state_prob[i][j];
    }
  }
  for(int i = 0; i < p.tau_states; i++) {
    for(int j = 0; j < p.nu_states; j++) {
      posterior_state_prob[i][j] /= state_prob_sum;
      // Rcpp::Rcout << posterior_state_prob[i][j] << ", ";
    }
    // Rcpp::Rcout << endl;
  }
  // Rcpp::Rcout << endl << endl;
  
  /*
  for(int i = 0; i < p.tau_states; i++) {
    for(int j = 0; j < p.nu_states; j++) {
      posterior_state_prob[i][i] = 1.0 / (p.tau_states * p.nu_states);
    }
  }*/
  
  for(int i = 0; i < p.tau_states; i++) {
    for(int j = 0; j < p.nu_states; j++) {
      for(int k = 0; k < pow(2,maxlevel); k++) {
        return_mean_density[k] += posterior_state_prob[i][j] * mean_density[i][j][k];
        for(int l = 0; l < g; l++) {
          return_sample_densities[l][k] += posterior_state_prob[i][j]  * sample_densities[i][j][l][k];
        }
        return_variance_function[k] += posterior_state_prob[i][j]  * (variance_function_term1[i][j][k] - variance_function_term2[i][j][k]);
      }
    }
  }
  
  Rcpp::List return_list = Rcpp::List::create(
    Rcpp::Named("data") = Rcpp::List::create(Rcpp::Named("value") = x, Rcpp::Named("group") = groups),
    Rcpp::Named("depth") = maxlevel,
    Rcpp::Named("mean_density") = Rcpp::wrap(return_mean_density),
    Rcpp::Named("sample_densities") = Rcpp::wrap(return_sample_densities),
    Rcpp::Named("variance_function") = Rcpp::wrap(return_variance_function),
    Rcpp::Named("logML") = Rcpp::wrap(return_log_ML)
  );
  return_list.attr("class") = "HAPT";
  return return_list;
}