# construct merged networks

# read all the matrices
msparcc <- read.csv(file="/content/CRCSparccNew.csv", header=FALSE)
mspieceasi <- read.csv(file="/content/SEglasso_CRC.csv", header=FALSE)
mmic <- read.csv(file="/content/micCRC.csv", header=FALSE)

# remove row/column
msparcc <- msparcc[-1,-1]
mspieceasi <- mspieceasi[-1,-1]
mmic <- mmic[-1,-1]

# according to paper, correlation threshhold > 0.2 forms edges in graph for MIC
mmic <- mmic >= 0.2
mmic <- apply(mmic, 2, as.integer)

# all of integer types
msparcc <- data.matrix(msparcc)
mspieceasi <- data.matrix(mspieceasi)

#normalizing after converting
msparcc[msparcc == 1] <- 0
msparcc[msparcc == 2] <- 1
mspieceasi[mspieceasi == 1] <- 0
mspieceasi[mspieceasi == 2] <- 1

# merge the networks
merge = abs(msparcc) + abs(mspieceasi) + abs(mmic)
merge[merge>1]<-1

write.csv(merged, "/content/S_SE_MIC.csv")