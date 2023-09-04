# data preparation
Source: https://gitlab.com/arcgl/rmagma
library(devtools)
install_gitlab("arcgl/rmagma")
library(rMAGMA)

dataset <- read.csv("/content/otu_IBD.csv")
#dataset <- read.csv("/content/abundance_CRC.csv")
dataset <-subset(dataset, select=-c(1,1))
dataset <- dataset[1:(length(dataset)-1)]
dataset

prevalence_measure = 0
prevalence <- colMeans(dataset>0)
sequencing_depth <- rowSums(dataset)
icol <- prevalence > prevalence_measure
irow <- sequencing_depth>500

col_list <- list()
for (x in 1:length(icol)) {
  if (isFALSE(icol[x])){
    col_list <- append(col_list, x)
  }
}
print(col_list)
write.csv(col_list, "/content/col_list.csv")
otu_table_formatted <- dataset[irow,icol]

write.csv(otu_table_formatted, "/content/otu_table_formatted.csv")
otu_table_formatted