monthly_mean_dir <- "/Volumes/SSD_2/Malaysia/02_Timeseries/CCM/01_region_mean/EVI"
out_dir <- "/Volumes/SSD_2/Malaysia/02_Timeseries/Nonliner"

# install.packages("zoo")
# install.packages("tidyverse")

# Load necessary libraries
library(mgcv)   # For GAM
library(dlnm)   # For DLNM modeling
library(splines) # For spline functions
library(zoo) # STL
library(tidyverse) #STL

# ----------------------------------------------
# Function to calculate STL window lengths based on Cleveland et al., 1990
# ----------------------------------------------
nt_calc <- function(f, ns) {
  nt <- (1.5 * f) / (1 - 1.5 * (1 / ns)) + 1
  nt <- ceiling(nt)  # round up
  if (nt %% 2 == 0) nt <- nt + 1
  return(nt)
}
nl_calc <- function(f) {
  f <- ceiling(f)
  if (f %% 2 == 0) f <- f + 1
  return(f)
}
robust_stl <- function(series, period = 12, smooth_length = 7) {
  # Calculate trend and low-pass window lengths
  trend_len <- nt_calc(period, smooth_length)
  lowpass_len <- nl_calc(period)
  
  # Apply STL decomposition
  # stl() requires a ts object
  ts_series <- ts(series, frequency = period)
  fit <- stl(ts_series, s.window = smooth_length, t.window = trend_len, robust = TRUE)
  return(fit)
}
# ----------------------------------------------


csvs <- list.files(path = monthly_mean_dir, pattern = "\\.csv$", full.names = TRUE)

for (cfile in csvs) {
  region <- strsplit(basename(cfile), "_")[[1]][1]
  df <- read.csv(cfile, row.names = 1)
  df$Date <- as.Date(rownames(df))
  df <- na.omit(df)
  
  # STL for EVI
  ser_evi <- df$EVI
  stl_evi <- robust_stl(ser_evi, period = 12, smooth_length = 7)
  plot(stl_evi)
  ser_resid <- stl_evi$time.series[, "remainder"] #residual
  
  # Normalize (z-score)
  df_numeric <- df[sapply(df, is.numeric)] 
  df_z <- as.data.frame(scale(df_numeric))
  ser_resid_z <- as.numeric(scale(ser_resid)) #this resid EVI used
  
  # Create crossbasis objects
  varnames <- colnames(df_z)
  varnames <- varnames[varnames != "EVI"]  # Exclude response variable
  
  # Set up logknots
  # lk <- logknots(c(0,24),3)
  lk <- c(6, 12, 18) # maybe 3 internal knots and 0,24 boundary knots. #no log scale for equal effect distribution
  # cb_list <- list()
  for (var in varnames) {
    xi <- df_z[[var]]
    cb <- crossbasis(xi, lag = c(0, 24), argvar = list(fun = "bs", degree = 2, df = 3),
                                         arglag = list(fun = "bs", degree = 2, df = 3, knots = lk))
    # cb_list[[var]] <- cb
    
    model <- gam(ser_resid_z ~ cb, family = gaussian(link = "identity"))
    print(summary(model))
    
    pred <- crosspred(cb,model,cumul=TRUE, cen=0) #cumlative lag effect
    pred_all <- crosspred(cb,model,cumul=FALSE, cen=0) #consider each lag effect 
    #allfit: vector https://rdrr.io/cran/dlnm/man/crosspred.html
    
    ### Plot the overall 3D exposure-lag-response surface
    xlab <- paste(var, "(standardized)")
    plot(pred, xlab=xlab, zlab = "Effect on EVI",ylab = "Lag (months)", phi = 30, theta = 230)
    plot(pred_all, xlab=xlab, zlab = "Effect on EVI",ylab = "Lag (months)", phi = 30, theta = 230)
    
    ### Plot the overall cumulative association
    redcum <- crossreduce(cb, model, type="overall", lag=24, cen=0)
    par(mai = c(1, 1, 0.25, 1), lwd = 3)
    lwdval =3
    # Plot pred
    plot<-plot(redcum,col=1,lty=1,ci.arg=list(col="Gainsboro"),xlab=xlab, ylab="Effect on EVI",cex.lab = 2.5, cex.axis = 2.5,lwd=lwdval)
    # horizontal reference line at y = 0
    # abline(h = 0, col = "darkgrey", lwd =lwdval)
    # Histgraom
    par(new = TRUE)
    par(mai = c(1, 1, 2, 1), lwd =lwdval)
    hist(df_z[[var]],
         col = rgb(240, 128, 128, 50, maxColorValue = 255),
         breaks = 15, axes = FALSE, ann = FALSE, xaxt = "n",
         cex.axis = 2.5, freq = FALSE, border = "black")
    # Add right-side (proportion) axis
    axis(side = 4, cex.axis = 2.5, yaxt = "n")
    axis(side = 4, at = c(0.0, 0.3, 0.6),
         labels = c(0.0, 0.3, 0.6), cex.lab = 2.5, cex.axis = 2.5, lwd = lwdval)
    
    dev.off()
    
    # ----------------------------------------------------------------
    ### threshold where the cumulative effect on EVI becomes negative
    ## need to revert value from z-score later
    # ----------------------------------------------------------------
    # Extract predictor values and cumulative effect
    var_vals <- pred$predvar
    effect_cum <- pred$cumfit  # or pred$allfit for non-cumulative (each lag)
    effect_all <- pred$allfit
    # Identify where effect is negative
    neg_indices_cum <- which(effect_cum < 0)
    neg_vals_cum <- var_vals[neg_indices_cum]
    neg_indices_all <- which(effect_all < 0)
    neg_vals_all <- var_vals[neg_indices_all]
    # threshold
    thre_cum_min <- min(neg_vals_cum, na.rm = TRUE)
    thre_all_min <- min(neg_vals_all, na.rm = TRUE)
    thre_cum_max <- max(neg_vals_cum, na.rm = TRUE)
    thre_all_max <- max(neg_vals_all, na.rm = TRUE)
    thre_min_fin = max(thre_cum_min, thre_all_min, na.rm=TRUE) #conservative
    thre_max_fin = min(thre_cum_max, thre_all_max, na.rm=TRUE)
    
    # -----------------------------------------
    ### Plot contour for lag and value association
    # -----------------------------------------
    
  }
  
  # dim(cb)
  # colnames(cb) 

  # ----------------
  # Test
  # single(できた)
  model_test <- gam(df_z$EVI ~ cb, family = gaussian(link = "identity"))
  # multiple(できた)
  cb_data <- do.call(cbind, cb_list)
  model_test <- gam(df_z$EVI ~ cb_data, family = gaussian(link = "identity"))
  print(summary(model_test))
  
  ## Can single variable
  pred <- crosspred(cb,model_test,cumul=TRUE,cen=0) #You can get lag result
  # Plot the overall 3D exposure-lag-response surface
  plot(pred, xlab = "Rain (standardized)", zlab = "Effect on EVI", 
       ylab = "Lag (weeks)", phi = 30, theta = 230)
  # ----------------  

  
  
  
  # Optionally save model or results
  # saveRDS(model, file = file.path(out_dir, paste0(region, "_dlnm_gam_model.rds")))
}
