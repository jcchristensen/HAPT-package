plot.HAPT <- function(obj, ylim = NULL, density_title = "", variance_title = "", cv_title = "", ...) {
  endpoints <- (1:(2^obj$depth-1))/2^obj$depth
  x <- c(0, rep(endpoints, each = 2), 1)
  mean_density <- rep(obj$mean_density, each = 2)
  if(is.null(ylim)) {
    ylim <- c(0, 2*max(obj$mean_density))
  }
  plot(x, mean_density, type='l', ylim = ylim, ylab = "Density", main = density_title, ...)
  for(i in 1:length(obj$sample_densities)) {
    lines(x, rep(obj$sample_densities[[i]], each = 2), col = "#c0c0c080")
  }
  lines(x, mean_density)
  
  # plot CV function
  cv <- rep(sqrt(obj$variance_function)/obj$mean_density, each = 2)
  plot(x, cv, ylim=c(0, 1.1*max(cv)), type='l', ylab = "CV function", main = cv_title, ...)
}
