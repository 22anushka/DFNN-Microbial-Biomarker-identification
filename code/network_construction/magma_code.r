# Source: https://gitlab.com/arcgl/rmagma

library(devtools)
install_gitlab("arcgl/rmagma")
library(rMAGMA)


# load table
otu_table_formatted = read.csv("/content/otu_table_formatted_10%.csv")
# remove label column
otu_table_formatted <- otu_table_formatted[,-1]

magma_ibd <- magma(data = otu_table_formatted)
write.csv(magma_ibd$refit, "/content/MAGMA_new.csv")