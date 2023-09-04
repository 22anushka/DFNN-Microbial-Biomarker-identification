
# Source: https://github.com/zdk123/SpiecEasi

library(devtools)
install_github("zdk123/SpiecEasi")
library(SpiecEasi)

dataset <- read.csv("/content/otu_table_formatted_10%.csv")
# to remove the header for column and row
dataset <-subset(dataset, select=-c(1,1))

# ensuring Label column is removed, else:
# to have data frame without label column
# X <- subset(dataset, select = -c(label))
# label column stored in Y
# Y <- as.data.frame(dataset$label)
# dataset <- X


# SPARCC
sparccibd <- sparcc(dataset)
# Define threshold for SparCC correlation matrix for the graph 
# taking 0.3 as threshold as defined in []
sparcc.graph <- abs(sparccibd$Cor) >= 0.3
diag(sparcc.graph) <- 0
sparccmatrix <- sparcc.graph
write.csv(sparccmatrix, "/content/SparccNew.csv")

#SPIEC EASI
spieceasiotu <- as.matrix(dataset)
# using default values and glasso method for chosen dataset
seibd <- spiec.easi(spieceasiotu, method='glasso', lambda.min.ratio=1e-2,
                          nlambda=20, pulsar.params=list(rep.num=5))
se_matrix <- getRefit(seibd)
se_matrix <- as.matrix(se_matrix)
write.csv(se_matrix, "/content/SEglasso.csv")

#MIC
install.packages("minerva")
library(minerva)

mic_result <- mine(dataset, C=10) # allowing all the other parameters to be default parameters
write.csv(mic_result$MIC, "/content/mic.csv")