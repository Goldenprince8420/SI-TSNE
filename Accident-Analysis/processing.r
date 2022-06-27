data <- Processed_data

library(missForest)

set.seed(81)
data_imp = missForest(data)

View(data_imp.ximp)

df <- data_imp[['ximp']]

save(df, file = "Downloads/saved.csv")

data <- driver_processed

View(data)

pairs(data)

pairs(~ data.V1_Lane_Nu + data.V2_Lane_Nu + data.Injured + data.Property_D)

plot(data)

pairs(data,                     # Data frame of variables
      labels = colnames(data),  # Variable names
      pch = 21,                 # Pch symbol
      main = "Driver Dataset",    # Title of the plot
      row1attop = TRUE,         # If FALSE, changes the direction of the diagonal
      gap = 1,                  # Distance between subplots
      cex.labels = NULL,        # Size of the diagonal text
      font.labels = 1)


data <- driver_readied









# Function to add histograms
panel.hist <- function(x, ...) {
  usr <- par("usr")
  on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5))
  his <- hist(x, plot = FALSE)
  breaks <- his$breaks
  nB <- length(breaks)
  y <- his$counts
  y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = rgb(0, 1, 1, alpha = 0.5), ...)
  # lines(density(x), col = 2, lwd = 2) # Uncomment to add density lines
}

# Creating the scatter plot matrix
pairs(data,
      upper.panel = NULL,         # Disabling the upper panel
      diag.panel = panel.hist)    # Adding the histograms















# Function to add correlation coefficients
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...) {
  usr <- par("usr")
  on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  Cor <- abs(cor(x, y)) # Remove abs function if desired
  txt <- paste0(prefix, format(c(Cor, 0.123456789), digits = digits)[1])
  if(missing(cex.cor)) {
    cex.cor <- 0.4 / strwidth(txt)
  }
  text(0.5, 0.5, txt,
       cex = 1 + cex.cor * Cor) # Resize the text by level of correlation
}

# Plotting the correlation matrix
pairs(data,
      upper.panel = panel.cor,    # Correlation panel
      lower.panel = panel.smooth) # Smoothed regression lines


















install.packages("gclus")
library(gclus)

# Correlation in absolute terms
corr <- abs(cor(data)) 

colors <- dmat.color(corr)
order <- order.single(pearson_r)

cpairs(pearson_r,                    # Data frame of variables
       order,                   # Order of the variables
       panel.colors = colors,   # Matrix of panel colors
       border.color = "grey70", # Borders color
       gap = 0.45,              # Distance between subplots
       main = "Ordered variables colored by correlation", # Main title
       show.points = TRUE,      # If FALSE, removes all the points
       pch = 21,                # pch symbol) # Colors by group

















install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)

chart.Correlation(data, histogram = TRUE, method = "pearson")


















install.packages("psych")
library(psych)

pairs.panels(data,
             smooth = TRUE,      # If TRUE, draws loess smooths
             scale = FALSE,      # If TRUE, scales the correlation text font
             density = TRUE,     # If TRUE, adds density plots and histograms
             ellipses = TRUE,    # If TRUE, draws ellipses
             method = "pearson", # Correlation method (also "spearman" or "kendall")
             pch = 21,           # pch symbol
             lm = FALSE,         # If TRUE, plots linear fit rather than the LOESS (smoothed) fit
             cor = TRUE,         # If TRUE, reports correlations
             jiggle = FALSE,     # If TRUE, data points are jittered
             factor = 2,         # Jittering factor
             hist.col = 4,       # Histograms color
             stars = TRUE,       # If TRUE, adds significance level with stars
             ci = TRUE)          # If TRUE, adds confidence intervals






library(psych)

corPlot(data, cex = 1.2)













install.packages("corrgram")
library(corrgram)

corrgram(data,
         order = TRUE,              # If TRUE, PCA-based re-ordering
         upper.panel = panel.pie,   # Panel function above diagonal
         lower.panel = panel.shade, # Panel function below diagonal
         text.panel = panel.txt,    # Panel function of the diagonal
         main = "Driver Correlogram")      # Main title

















install.packages('corrplot')
library(corrplot)

corrplot(cor(data),        # Correlation matrix
         method = "shade", # Correlation plot method
         type = "full",    # Correlation plot style (also "upper" and "lower")
         diag = TRUE,      # If TRUE (default), adds the diagonal
         tl.col = "black", # Labels color
         bg = "white",     # Background color
         title = "",       # Main title
         col = NULL)       # Color palette














par(mfrow = c(2, 3))

# Circles
corrplot(cor(data), method = "circle",
         title = "method = 'circle'",
         tl.pos = "n", mar = c(2, 1, 3, 1)) 
# Squares 
corrplot(cor(data), method = "square",
         title = "method = 'square'",
         tl.pos = "n", mar = c(2, 1, 3, 1)) 
# Ellipses
corrplot(cor(data), method = "ellipse",
         title = "method = 'ellipse'",
         tl.pos = "n", mar = c(2, 1, 3, 1)) 
# Correlations
corrplot(cor(data), method = "number",
         title = "method = 'number'",
         tl.pos = "n", mar = c(2, 1, 3, 1)) 
# Pie charts
corrplot(cor(data), method = "pie",
         title = "method = 'pie'",
         tl.pos = "n", mar = c(2, 1, 3, 1)) 
# Colors
corrplot(cor(data), method = "color",
         title = "method = 'color'",
         tl.pos = "n", mar = c(2, 1, 3, 1)) 

par(mfrow = c(1, 1))



















corrplot(cor(data),
         method = "circle",       
         order = "hclust",         # Ordering method of the matrix
         hclust.method = "ward.D", # If order = "hclust", is the cluster method to be used
         addrect = 2,              # If order = "hclust", number of cluster rectangles
         rect.col = 3,             # Color of the rectangles
         rect.lwd = 3)             # Line width of the rectangles























# install.packages("corrplot")
library(corrplot)

corrplot.mixed(cor(data),
               lower = "shade", 
               upper = "circle",
               tl.col = "black",
               title = "corrplot_mized")














# _______________________Plot of Correlation Analysis____________________________ #
data_pearson <- pearson_r

pearson_r$column <- unlist(pearson_r$column)





# Pearson Correlation
corrplot(data.matrix(pearson_r),        # Correlation matrix
         method = "shade", # Correlation plot method
         type = "full",    # Correlation plot style (also "upper" and "lower")
         is_corr = FALSE,
         order = 'hclust',
         diag = TRUE,      # If TRUE (default), adds the diagonal
         tl.col = "black", # Labels color
         bg = "white",     # Background color
         title = "Correlation Plot Pearson",       # Main title
         mar = c(2, 1, 3, 1),
         col = NULL)       # Color palette





# Spearman Correlation
corrplot(data.matrix(spearman_r), 
         method = "circle",
         title = "Correlation Plot Spearman",
         type = 'full',
         is_corr = FALSE,
         order = 'FPC',
         diag = TRUE,
         bg = "white",
         col = COL2('PiYG'),
         mar = c(2, 1, 3, 1),
         t1.col = "black") 



# Point Biserial Correlation
corrplot(data.matrix(point_biserial_r), 
         method = "ellipse",
         title = "Correlation Plot Point Biserial",
         type = 'full',
         is_corr = FALSE,
         order = 'hclust',
         diag = TRUE,
         bg = "white",
         col = COL2('PuOr', 20),
         mar = c(2, 1, 3, 1),
         t1.col = "black")




# Kendal Tau Correlation
corrplot(data.matrix(kendal_tau_r), 
         method = "square",
         title = "Correlation Plot Kendal Tau",
         is_corr = FALSE,
         type = 'upper',
         order = 'AOE',
         diag = TRUE,
         bg = "white",
         col = COL2('RdYlBu', 100),
         mar = c(2, 1, 3, 1),
         t1.col = "black")





# Weighted Tau Correlation
corrplot(data.matrix(weighted_tau_r), 
         method = "square",
         title = "Correlation Plot Weighted Tau",
         is_corr = FALSE,
         type = 'lower',
         order = 'hclust',
         diag = TRUE,
         bg = "white",
         col = COL2(n=200),
         mar = c(2, 1, 3, 1),
         t1.col = "black")




## Linear Regression Correlation
corrplot(cor(lin_reg_slope), 
         method = "color",
         title = "Correlation Plot Linear Regression",
         is_corr = FALSE,
         type = 'lower',
         order = 'hclust',
         diag = TRUE,
         bg = "white",
         col = COL2('PRGn'),
         mar = c(2, 1, 3, 1),
         t1.col = "black")
















