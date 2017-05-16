### R code
### Paper " A dynamic factor machine learning method for multi-variate and multi-step-ahead forecasting"
### Author: Gianluca Bontempi
### ULB, Machine Learning Group, mlg.ulb.ac.be


## Experimental comparison on temperature data of the following forecasting algorithms

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

rm(list=ls())



library("psych")
require(gbcode)
require(autoencoder)
require(rnn)
require(pls)
require(MTS)
library(kerasR)
library(Rssa)


source("nar.R")
source("starpred.R")
source("rescale.R")

N<-2000
n<-50
p0=3
m=2
CC=2
savefile<-TRUE
models=c("lazyiter","lazydirect","mimo.comb")

namesAlgos=c(" DFML_PC "," DFML'_PC "," DFM "," DFML_A "," DFML'_A "," RNN "," DSE "," PLS "," UNI "," VAR "," SSA ")
Nval=20
Algos=setdiff(1:9,7)



for (n in c(100,200)){ ## number of countries
  allS=NULL
  allE=NULL
  allE2=NULL
  allE3=NULL
  allE4=NULL
  allE5=NULL
  allE6<-NULL
  allE7<-NULL
  allE8<-NULL
  allE9<-NULL
  allE10<-NULL
  allE11<-NULL
  allE12<-NULL
  allE0<-NULL
 
  for (number in seq(1,2,by=0.2)){ # controls the size of training set
    for (H in c(2,5,10,20,50)){ ## horizons
      set.seed(number+n+H)
      namefile=paste("./data/bench.temp",n,"Rdata",sep=".")
      
      load(namefile)
      X=Temp
      if (NROW(X)>N)
        X=Temp[1:N,]
      
      A=ts(X,freq=12)
      decomposeA=stats::decompose(A,"additive")
      X=A-decomposeA$seasonal
      
      X=scale(X)
      N=NROW(X)
      
      
      Ntr=round(number*N/3)
      Xtr=X[1:Ntr,]
      
      
      if (is.element(2,Algos)){
        cat("D: pcadesign \n")
        ptm <- proc.time()
        P=pcadesign(Xtr,2*m,H,2*p0,CC=2*CC)
        cat("Elapsed pcadesign=",proc.time() - ptm,"\n")
      }
      
      if (is.element(4,Algos)){
        cat("\n -- \n D: autodesign \n")
        ptm <- proc.time()
        A=autodesign(Xtr,m,H,p0,CC=CC,calibrate=FALSE)
        cat("Elapsed autodesign=",proc.time() - ptm,"\n")
      }
      
      if (is.element(5,Algos)){
        cat("\n -- \n D: autodesign \n")
        ptm <- proc.time()
        A2=autodesign(Xtr,2*m,H,2*p0,CC=2*CC,calibrate=TRUE)
        cat("Elapsed autodesign2=",proc.time() - ptm,"\n")
      }
      
      if (is.element(6,Algos)){
        cat("\n -- \n D: rnndesign \n")
        ptm <- proc.time()
        ##R=rnndesign(Xtr,m,H,p0,cal=FALSE)
        R=kerasdesign(Xtr,m,H,p0,cal=FALSE)
        cat("Elapsed rnndesign=",proc.time() - ptm,"\n")
      }
      
      if (is.element(7,Algos)){
        cat("\n -- \n D: dsedesign \n")
        ptm <- proc.time()
        D=dsedesign(Xtr,m,H)
        cat("Elapsed dsedesign=",proc.time() - ptm,"\n")
      }
      
      if (is.element(8,Algos)){
        cat("\n -- \n D: plsdesign \n")
        ptm <- proc.time()
        PL=plsdesign(Xtr,2*m,H,2*p0)
        cat("Elapsed plsdesign=",proc.time() - ptm,"\n")
        
      }
      
      
      if (is.element(9,Algos)){
        ptm <- proc.time()
        U=unidesign(Xtr,m,H)
        cat("Elapsed unidesign=",proc.time() - ptm,"\n")
      }
      
      if (is.element(10,Algos)){
        ptm <- proc.time()
        V=vardesign(Xtr,1)
        cat("Elapsed vardesign=",proc.time() - ptm,"\n")
      }
      
      
      for (s in seq(Ntr,N-H,length.out=Nval)){
        
        Itr<-1:s
        Xtr=X[1:s,]
        Xts=X[(s+1):(s+H),]
        
        Xhat0=naivepred(Xtr,H)
        
####### ALGO 1
        Xhat=Xhat0
        if (is.element(1,Algos)){
          Xhat=pcapred(Xtr,m,H,p0,cc=CC,mod=models[1])
        }
####### ALGO 2
        Xhat2=Xhat0
        if (is.element(2,Algos)){
          Xhat2=pcapred(Xtr,P$m,H,P$p,cc=P$cc,mod=P$mod,V=P$V)
        }
####### ALGO 3
        Xhat3=Xhat0
        if (is.element(3,Algos)){
          Xhat3=pcapredse(Xtr,m,H,p0)
        }
####### ALGO 4
        Xhat4=Xhat0
        if (is.element(4,Algos)){
          Xhat4= autopred(Xtr,A$m,H,A$p,cc=A$cc,mod=models[1],encoder=A$encoder)
        }
        
####### ALGO 5
        Xhat5=Xhat0
        if (is.element(5,Algos)){
          Xhat5= autopred(Xtr,A2$m,H,A2$p,cc=A2$cc,mod=A2$mod,encoder=A2$encoder)
        }
######## ALGO 6
        Xhat6=Xhat0
        if (is.element(6,Algos)){
        Xhat6=keraspred(Xtr,R,m, H)
        ## Xhat6=rnnpred(Xtr,R,m, H)
        }
        
######## ALGO 7
        Xhat7=Xhat0
        if (is.element(7,Algos)){
          Xhat7=dsepred(Xtr,m,H,D$mod)
        }
######## ALGO 8
        Xhat8=Xhat0
        if (is.element(8,Algos)){
          Xhat8=plspred(Xtr,PL$m,H,mod=PL$mod)
        }
######## ALGO 9
        Xhat9=Xhat0
        if (is.element(9,Algos)){
          Xhat9=unipred(Xtr,m,H,Emb=U$Emb,fsel=U$fsel)
        }
        
######## ALGO 10
        Xhat10=Xhat0
        if (is.element(10,Algos)){
          Xhat10=varpred(Xtr,H,V$m)
        }
######## ALGO 11
        Xhat11=Xhat0
        if (is.element(11,Algos)){
          Xhat11=ssapred(Xtr,H)
        }      
        e.hat=apply((Xts-Xhat)^2,2,mean)
        e.hat2=apply((Xts-Xhat2)^2,2,mean)
        e.hat3=apply((Xts-Xhat3)^2,2,mean)
        e.hat4=apply((Xts-Xhat4)^2,2,mean)
        e.hat5=apply((Xts-Xhat5)^2,2,mean)
        e.hat6=apply((Xts-Xhat6)^2,2,mean)
        e.hat7=apply((Xts-Xhat7)^2,2,mean)
        e.hat8=apply((Xts-Xhat8)^2,2,mean)
        e.hat9=apply((Xts-Xhat9)^2,2,mean)
        e.hat10=apply((Xts-Xhat10)^2,2,mean)
        e.hat11=apply((Xts-Xhat11)^2,2,mean)
        e.hat0=apply((Xts-Xhat0)^2,2,mean)
        allE=rbind(allE,c(H,number,n,mean(e.hat)))
        allE2=rbind(allE2,c(H,number,n,mean(e.hat2)))
        allE3=rbind(allE3,c(H,number,n,mean(e.hat3)))
        allE4=rbind(allE4,c(H,number,n,mean(e.hat4)))
        allE5=rbind(allE5,c(H,number,n,mean(e.hat5)))
        allE6=rbind(allE6,c(H,number,n,mean(e.hat6)))
        allE7=rbind(allE7,c(H,number,n,mean(e.hat7)))
        allE8=rbind(allE8,c(H,number,n,mean(e.hat8)))
        allE9=rbind(allE9,c(H,number,n,mean(e.hat9)))
        allE10=rbind(allE10,c(H,number,n,mean(e.hat10)))
        allE11=rbind(allE11,c(H,number,n,mean(e.hat11)))
        allE0=rbind(allE0,c(H,number,n,mean(e.hat0)))
        
        cat(".")
        
      } ## for s
      
##### PRINT OUT OF RESULTS
      
      indH<-which(allE0[,1]==H  & allE0[,3]==n)
      cat("\n **** \n ","number countries=",n,"number=",number, " H=",H)
      cat( "Naive :",mean(allE0[indH,4]))
      if (is.element(1,Algos))
        cat( namesAlgos[1],":",mean(allE[indH,4]))
      if (is.element(2,Algos))
        cat(namesAlgos[2],":",mean(allE2[indH,4]))
      if (is.element(3,Algos))
        cat(namesAlgos[3],":",mean(allE3[indH,4]))
      if (is.element(4,Algos))
        cat(namesAlgos[4],":",mean(allE4[indH,4]))
      if (is.element(5,Algos))
        cat(namesAlgos[5],":",mean(allE5[indH,4]))
      if (is.element(6,Algos))
        cat(namesAlgos[6],":",mean(allE6[indH,4]))
      if (is.element(7,Algos))
        cat(namesAlgos[7],":",mean(allE7[indH,4]))
      if (is.element(8,Algos))
        cat(namesAlgos[8],":",mean(allE8[indH,4]))
      if (is.element(9,Algos))
        cat(namesAlgos[9],":",mean(allE9[indH,4]))
      if (is.element(10,Algos))
        cat(namesAlgos[10],":",mean(allE10[indH,4]))
      if (is.element(11,Algos))
        cat(namesAlgos[11],":",mean(allE11[indH,4]))
      cat("\n")
      
      
      
    }## for H
  }  ## for number
  
##### SAVING RESULTS
  namefile=paste("./results/tempbench.d",n,"Rdata",sep=".")
  if (savefile)
    save(file=namefile,list=c("Algos","allE","allE2","allE3","allE4","allE5","allE6","allE7","allE8","allE9","allE10","allE11","allE0"))
  
} ## for n
