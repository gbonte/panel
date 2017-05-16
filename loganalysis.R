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

for (n in c(20,50,100,200,400,1000)){

  load(paste("./results/bench",n,"Rdata",sep="."))
  nAlgos=c( "DFMLPC", "DFML'PC", "DFM", "DFMLA", "DFML'A", "RNN", "DSE", "PLS", "UNI", "VAR",  "NAIVE")
  
  pv=numeric(length(nAlgos))
  
  
  Algos=1:11
  showOrder<-c(3,1,2,4,5,6,7,8,9,10,11)
  IT<-c(0,10,20)
  for (H in c(5,10,20)){
    IH=which(allE[,2]==H)
    E<-cbind(allE[IH,5], allE2[IH,5],allE3[IH,5], allE4[IH,5], allE5[IH,5],
             allE6[IH,5], allE7[IH,5], allE8[IH,5], allE9[IH,5], allE10[IH,5],
             allE0[IH,5])
    
    colnames(E)<-nAlgos[Algos]
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




