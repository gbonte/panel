rm(list=ls())


## 0: naive
## 1:  DFML: PCA + KNN
## 2:  DFML: PCA  + KNN calibrated
## 3:  DFM: PCA +DSE
## 4:  DFML: AUTOENCODE  + KNN
## 5:  DFML: AUTOENCODE  + KNN calibrated
## 6:  RNN
## 7:  DSE
## 8:  PLS
## 9:  UNI
## 10: VAR
## 11: SSA

for (n in c(100,200)){

  load(paste("./results/tempbench.d",n,"Rdata",sep="."))
  
  
  nAlgos=c( "DFMLPC", "DFML'PC", "DFM", "DFMLA", "DFML'A", "RNN", "DSE", "PLS", "UNI", "VAR",  "NAIVE")
  
  pv=numeric(length(nAlgos))
  
  
  showOrder<-c(3,1,2,4,5,6,8,9,11)
  IT<-c(0,10,20)
  for (H in c(2,5,10,20,50)){
    IH=which(allE[,1]==H)
    E<-cbind(allE[IH,4], allE2[IH,4],allE3[IH,4], allE4[IH,4], allE5[IH,4],
             allE6[IH,4], allE7[IH,4], allE8[IH,4], allE9[IH,4], allE10[IH,4],
             allE0[IH,4])
    
    colnames(E)<-nAlgos
    best=which.min(apply(E,2,mean,na.rm=TRUE))
    allbest=E[,best]
    
    for (i in 1:NCOL(E)){
      if (i !=best)
        pv[i]=t.test(E[,i],E[,best],paired=TRUE)$p.value
    }
    pv[best]=0
    
    
    cat("n=",n,"H=",H,"\n")
    print(round(apply(E[,showOrder],2,mean,na.rm=TRUE),3))
    cat("best=", nAlgos[best], "pv=",nAlgos[which(pv>0.05)],"\n")
    browser()
    
  }
  
  
}

