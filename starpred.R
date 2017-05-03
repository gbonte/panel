require(dse)

Kmin=3

MakeEmbeddedrev<-function(ts, n, delay, hor = 1, w = 1) 
{
    no.data <- NROW(ts)
    no.var <- NCOL(ts)
    a <- NROW(n)
    b <- NCOL(n)
    if (a != no.var) 
        stop("Error in the size of embedding n")
    if (length(delay) != no.var) 
        stop("Error in the size of delay")
    if (length(hor) != length(w)) 
        stop("Error in the size of horizon hor")
    N <- no.data - max(n) - max(delay)
    Input <- array(0, c(N, sum(n)))
    Output <- array(0, c(N, sum(hor)))
    for (i in 1:N) {
        for (j in 1:no.var) {
            k <- 1:n[j]
            Input[i, sum(n[1:j - 1]) + k] <- rev(ts[i + n[j] - k + 
                max(n) - n[j] + max(delay) - delay[j], j])
            for (ww in 1:length(w)) {
                if (ww == 1) 
                  iw <- 0
                else iw <- sum(hor[1:(ww - 1)])
                Output[i, (iw + 1):(sum(hor[1:ww]))] <- numeric(hor[ww]) + 
                  NA
                M <- min(no.data, (i + max(n) + max(delay) + 
                  hor[ww] - 1))
                Output[i, (iw + 1):(iw + M - (i + max(n) + max(delay)) + 
                  1)] <- ts[(i + max(n) + max(delay)):M, w[ww]]
            }
        }
    }
    list(inp = Input, out = Output)
}

########### DSE 
dsedesign<-function(X,m0,H){ 
  maxm=round(2*m0) 
  n<-NCOL(X)
  N<-NROW(X)
  Xtr<-X[1:round(2*N/3),]
  Ntr<-NROW(Xtr)
  Xts<-X[(Ntr+1):N,]
  Nts<-NROW(Xts)
  
  Ehat<-numeric(maxm)
  
  for (m in 1:maxm){
    GG<-TSdata(output=Xtr)
    model <- estBlackBox(GG,max.lag=m,verbose=FALSE)
    for (s in seq(1,Nts-H,length.out=10)){
      Xts=X[(Ntr+s+1):(Ntr+s+H),]
      pr <- dse::forecast(model,hor=H)$forecast
      Xhat=array(pr[[1]],c(H,n))
      Ehat[m]<-Ehat[m]+mean(apply((Xts-Xhat)^2,2,mean))
    }
  } 
  
  GG<-TSdata(output=X)
  m<-which.min(Ehat)
  mod= estBlackBox(GG,max.lag=m,verbose=FALSE)
  return (list(mod=mod))
}


dsepred<-function(Xtr,m,H,model=NULL){
  n<-NCOL(Xtr)
  GG<-TSdata(output=Xtr)
  if (is.null(model))
    model <- estBlackBox(GG,max.lag=m,verbose=FALSE)
  pr <- dse::forecast(model,hor=H,data=GG)$forecast
  return(array(pr[[1]],c(H,n)))
}

########### KERAS


kerasdesign<-function(Xtr,m,H,p0,calibrate=TRUE){
  n=NCOL(Xtr)
  
  M=MakeEmbeddedrev(Xtr,numeric(n)+m,numeric(n),numeric(n)+H,1:n)
  I=which(!is.na(apply(M$out,1,sum)))
  M$inp=M$inp[I,]
  M$out=M$out[I,]
  N=NROW(M$inp)
  n1=NCOL(M$inp)
  p<-NCOL(M$out)

  trainX=array(M$inp,c(NROW(M$inp),m,n))
  trainY=M$out
  lmodel=list()
  model = Sequential()
  
  model$add(SimpleRNN(10,input_shape=c(m,n)))
  model$add(Dense(ncol(trainY)))
  
  keras_compile(model,loss='mse', optimizer=RMSprop())
  keras_fit(model,trainX, trainY, epochs=100,verbose=0)
  lmodel=model

  if (FALSE){
    for (i in 1:n){
      print(i)
      ind=((i-1)*H+1):(i*H)
      
      
      model$add(SimpleRNN(2,input_shape=c(m,n)))
      model$add(Dense(length(ind)))
                                        #model$add(Activation("relu"))
      
      keras_compile(model,loss='mse', optimizer=RMSprop())
      keras_fit(model,trainX, trainY[,ind], epochs=100,verbose=0)
      lmodel[[i]]=model
    }
  }
  
  return(lmodel)
  
}

keraspred<-function(Xtr,lmodel,m,H){
  q<-NULL
  N<-NROW(Xtr)
  n<-NCOL(Xtr)
  D=0
  for (j in 1:n)
        q<-c(q,rev(Xtr[seq(N-D,N-m+1-D,by=-1),j]))
  
  Xts=array(q,c(1,1,length(q)))
  trainXts=array(Xts,c(1,m,n))
  Yhat<-array(NA,c(H,n))
  Yhat  <- keras_predict(lmodel, trainXts,verbose=0)
  return(array(Yhat,c(H,n)))
  if (FALSE){
    for (i in 1:n){
      ind=((i-1)*H+1):(i*H)
      Yhat[,i]  <- keras_predict(lmodel[[i]], trainXts,verbose=0)
    }
    return(Yhat)
  }


}
  
########### RNN

rnndesign<-function(Xtr,m,H,p0,calibrate=TRUE){
  n=NCOL(Xtr)
  
  M=MakeEmbedded(Xtr,numeric(n)+m,numeric(n),numeric(n)+H,1:n)
  I=which(!is.na(apply(M$out,1,sum)))
  M$inp=M$inp[I,]
  M$out=M$out[I,]
  N=NROW(M$inp)
  n1=NCOL(M$inp)

  Ntr=N-H
  ## Create 3d array: dim 1: samples; dim 2: time; dim 3: variables.
  X=array(M$inp,c(1,N,n1))
  Y=array(M$out,c(1,N,NCOL(M$out)))

  Xtr=array(M$inp[1:Ntr,],c(1,Ntr,n1))
  maxp=min(n1,p0)
  Ytr=array(M$out[1:Ntr,],c(1,Ntr,NCOL(M$out)))

  Xts=array(M$inp[(1+Ntr):N,],c(1,H,n1))
  
  Yts=array(M$out[(1+Ntr):N,],c(1,H,NCOL(M$out)))

  if (calibrate){
    E=numeric(maxp)+Inf
    for (h in 2:(maxp)){
      model <- trainr(Y=Ytr, X=Xtr,learningrate= 0.01,
                      hidden=h,numepochs=10,network="lstm")
      Yhat  <- predictr(model,Xts)
      E[h]=0
      
      for (j in 1:dim(Yhat)[2])
        E[h]=E[h]+mean((Yts[,j,]-Yhat[,j,])^2)
      
    }
    p0=which.min(E)
  }
      
  model <- trainr(Y=Y, X=X,learningrate= 0.01,hidden_dim =p0,numepochs=50,network="lstm")
  return(model)
  
}


rnnpred<-function(Xtr,model,m,H){
  q<-NULL
  N<-NROW(Xtr)
  n<-NCOL(Xtr)
  D=0
  for (j in 1:n)
        q<-c(q,Xtr[seq(N-D,N-m+1-D,by=-1),j])
  
  Xts=array(q,c(1,1,length(q)))
  Yhat  <- predictr(model,Xts)
  
  return(array(Yhat,c(H,n)))

}

########### PCA
cor.prob <- function(X, dfr = nrow(X) - 2) {
  n<-NCOL(X)
  nt=n*(n-1)/2
  R <- cor(X)
  above <- row(R) < col(R)
  r2 <- R[above]^2
  Fstat <- r2 * dfr / (1 - r2)
  R[above] <- min(1,(1 - pf(Fstat, 1, dfr))*nt)
  R
}




pcadesign<-function(X,m0,H,p0,CC,lambda=0,Lcv=10){
  
  n<-NCOL(X)
  maxp=min(n,p0)
  maxm=m0
  nm=length(models)
  
  N<-NROW(X)
  Xtr<-X[1:round(2*N/3),]
  Ntr<-NROW(Xtr)
  Xts<-X[(Ntr+1):N,]
  Nts<-NROW(Xts)
  C=cov(Xtr) 
  V=t(eigen(C,TRUE)$vectors[,1:maxp])
  Z=X%*%t(V)


  
  Ehat<-array(0,c(CC,maxm,maxp,nm))
  for (mm in 1:nm){
    mod=models[mm]
    for (cc in 1:CC){
      for (m in 1:maxm){
        ZZ<-NULL
        for (s in seq(1,Nts-H,length.out=Lcv)){
          Zhat<-array(NA,c(H,maxp))
         
          Zhat[,1]=multiplestepAhead(Z[1:(Ntr+s),1],n=m, H=H,method=mod,Kmin=Kmin,C=cc)
          Xts=X[(Ntr+s+1):(Ntr+s+H),]
          Xhat=Zhat[,1]%*%array(V[1,],c(1,n))

          
          
          Ehat[cc,m,1,mm]<-Ehat[cc,m,1,mm]+mean(apply((Xts-Xhat)^2,2,mean))

          for (p in 2:maxp){
            Zhat[,p]=multiplestepAhead(Z[1:(Ntr+s),p],n=m, H=H,method=mod,Kmin=Kmin,C=cc)
            Xhat=Zhat[,1:p]%*%V[1:p,]
            Ehat[cc,m,p,mm]<-Ehat[cc,m,p,mm]+mean(apply((Xts-Xhat)^2,2,mean))
          } ## for p
          ZZ<-rbind(ZZ,Zhat)
        } ## for s
        if (lambda>0)
          for (p in 2:maxp){
            cZ=1-cor.prob(ZZ[,1:p])
            cZ= mean(c(cZ[upper.tri(cZ)]))
            Ehat[cc,m,p,mm]<-Ehat[cc,m,p,mm]/Lcv+lambda*cZ ## criterion for decorrelation of factor predictions 
          }
      } ## for m
    }
      cat(".")
  }

  Emin=min(Ehat)
  p0<-which.min(apply(Ehat,3,min))
  m<-which.min(apply(Ehat,2,min))
  cc<-which.min(apply(Ehat,1,min))
  mod=models[which.min(apply(Ehat,4,min))]
  C=cov(X)
  V=t(eigen(C,TRUE)$vectors[,1:p0])
  return (list(p=p0,m=m,cc=cc,mod=mod,V=V[1:p0,]))
}

pcapredse<-function(Xtr,m,H,p0,V=NULL){
  
  
  n<-NCOL(Xtr)
  if (is.null(V)){
    C=cov(Xtr)
    V=t(eigen(C,TRUE)$vectors[,1:p0])
  }
  
  V=array(V,c(p0,n))
  Ztr=Xtr%*%t(V)
  n<-NCOL(Xtr)
  GG<-TSdata(output=Ztr)
  
  model <- estBlackBox(GG,max.lag=m,verbose=FALSE)
  pr <- dse::forecast(model,hor=H,data=GG)$forecast
  return(array(pr[[1]],c(H,p0))%*%V[1:p0,])

  
  
  
}

normm<-function(u){
  return(as.numeric(u%*%u))

}
proj<-function(u,v){

  return(u*as.numeric(u%*%v)/as.numeric(u%*%u))

}

pcapred<-function(Xtr,m,H,p0,cc=2,mod,V=NULL,orth=FALSE){
  
  
  n<-NCOL(Xtr)
  if (is.null(V)){
    C=cov(Xtr)
    V=t(eigen(C,TRUE)$vectors[,1:p0])
  }
  
  V=array(V,c(p0,n))
  Ztr=Xtr%*%t(V)
  
  
  Zhat<-array(NA,c(H,p0))
  Zhat[,1]=multiplestepAhead(Ztr[,1],n=m, H=H,method=mod,Kmin=Kmin,C=cc)
  
  
  if (p0>1)
    for (p in 2:p0)
      Zhat[,p]=multiplestepAhead(Ztr[,p],n=m, H=H,method=mod,Kmin=Kmin,C=cc)

  U=Zhat

  if (orth & p0>1){## Gram-Schmidt orthogonalization
    ZZ=rbind(Ztr[,1:p0],Zhat[,1:p0]) 
    U=array(NA,c(NROW(ZZ),p0))
    
    U[,1]=ZZ[,1]
    
    for (i in 2:p0){
      p<-numeric(NROW(ZZ))
      for (j in 1:(i-1))
        p=p+proj(U[,j],ZZ[,i])
      U[,i]=ZZ[,i]-p
      
    }
    
    U=U[(NROW(Ztr)+1):NROW(U),]
  }
  if (p0>1)
    return(U[,1:p0]%*%V[1:p0,])
  return(U[,1]%*%array(V[1:p0,],c(1,n)))
  
  
}


########### VAR
vardesign<-function(X,m0){
 return(list(m=VAR(X,p=m0,output=FALSE)))

}

varpred<-function(X,H,m){
   #m1=VAR(X,p=m0,output=FALSE)
   return(VARpred2(X,m,H,Out=FALSE)$pred)

 }




 
########### AUTO

autodesign<-function(X,m0,H,p0,CC,calibrate=TRUE){
  n<-NCOL(X)
  maxm=m0
  maxp=min(n,p0)
  
  N<-NROW(X)
  Xtr<-X[1:round(2*N/3),]
  Ntr<-NROW(Xtr)
  Xts<-X[(Ntr+1):N,]
  
  rho=0.1
  lambda=0.001
  beta=5
  epsilon=0.01

  if (!calibrate){
    encoder<-autoencode(Xtr,nl=3,N.hidden=p0,lambda=lambda,
                        unit.type='tanh', beta=beta, rho=rho,epsilon=epsilon,
                        max.iterations=50000,rescale.flag=TRUE)
    return (list(m=m0,cc=CC,p=p0,encoder=encoder))
  }
  Nts<-NROW(Xts)
  nm=length(models)
  Ehat<-array(0,c(CC,maxm,maxp,nm))
  for (mm in 1:nm){
    mod=models[mm]
    for (p0 in 1:maxp){
      encoder<-autoencode(Xtr,nl=3,N.hidden=p0,lambda=lambda,
                          unit.type='tanh', beta=beta, rho=rho,epsilon=epsilon, max.iterations=5000,rescale.flag=TRUE)
    
    Z=predict(encoder,X,hidden.output=TRUE)$X.output
    for (cc in 1:CC){
      for (m in 1:maxm){
        for (s in seq(1,Nts-H,length.out=10)){
          Zhat<-array(NA,c(H,p0))
          Zhat[,1]=multiplestepAhead(Z[1:(Ntr+s),1],n=m, H=H,method=mod,Kmin=Kmin,C=cc)
          Xts=X[(Ntr+s+1):(Ntr+s+H),]
          if (p0>1)
            for (p in 2:p0)
              Zhat[,p]=multiplestepAhead(Z[1:(Ntr+s),p],n=m, H=H,method=mod,Kmin=Kmin,C=cc)
          
          Xhat=decode(encoder,Zhat)
          Ehat[cc,m,p0,mm]<-Ehat[cc,m,p0,mm]+mean(apply((Xts-Xhat)^2,2,mean))
        }
        
      }
     
    }
  }
    cat(".")
  }
  Emin=min(Ehat)
  m<-which.min(apply(Ehat,2,min))
  cc<-which.min(apply(Ehat,1,min))
  p0<-which.min(apply(Ehat,3,min))
  mod<-models[which.min(apply(Ehat,4,min))]
  encoder<-autoencode(X,nl=3,N.hidden=p0,lambda=lambda,
                      unit.type='tanh', beta=beta, rho=rho,epsilon=epsilon, max.iterations=50000,rescale.flag=TRUE)

  
  return (list(m=m,cc=cc,p=p0,mod=mod,encoder=encoder))
}

autopred<-function(Xtr,m,H,p0,cc,mod="iter",encoder=NULL){
 
  n<-NCOL(X)
 
 

  if (is.null(encoder)){
    rho=0.1
    lambda=0.001
    beta=5
    epsilon=0.01
    encoder<-autoencode(Xtr,nl=3,N.hidden=p0,lambda=lambda,
                        unit.type='tanh', beta=beta, rho=rho,epsilon=epsilon,
                        max.iterations=50000,rescale.flag=TRUE)
  }
  
  Ztr=predict(encoder,Xtr,hidden.output=TRUE)$X.output
  
  
  Zhat<-array(NA,c(H,p0))
  Zhat[,1]=multiplestepAhead(Ztr[,1],n=m, H=H,method=mod,Kmin=Kmin,C=cc)
 
  if (p0>1)
    for (p in 2:p0)
      Zhat[,p]=multiplestepAhead(Ztr[,p],n=m, H=H,method=mod,Kmin=Kmin,C=cc)
        
 
  return(decode(encoder,Zhat))
  
}

########### PLS
plsdesign<-function(Xtr,m0,H,p0){
  maxm=m0
  n<-NCOL(Xtr)
  N<-NROW(Xtr)
  maxp=min(n,p0)
  Xtr<-X[1:round(2*N/3),]
  Ntr<-NROW(Xtr)
  Xts<-X[(Ntr+1):N,]
  Nts<-NROW(Xts)
  Ehat<-array(0,c(maxm,maxp))
  for (m in 1:maxm){
    M=MakeEmbedded(Xtr,numeric(n)+m,delay=numeric(n),hor=numeric(n)+H,w=1:n)   
    for (p in 1:maxp){
      mod <- mvr(out ~ inp, ncomp = p, data = M)
      for (s in seq(1,Nts-H,length.out=10)){
        q<-NULL
        for (j in 1:n){
          for (h in seq(1:m))
            q<-c(q,X[Ntr+s-h+1,j])
        }
        Xts=X[(Ntr+s+1):(Ntr+s+H),]
        yhat<-predict(mod,array(q,c(1,length(q))))
        Xhat=array(yhat,c(H,n))
        Ehat[m,p]<-Ehat[m,p]+mean(apply((Xts-Xhat)^2,2,mean))
      }
     
    }
    cat(".")
  }

  Emin=min(Ehat)
  m<-which.min(apply(Ehat,1,min))
  p0<-which.min(apply(Ehat,2,min))
  M=MakeEmbedded(Xtr,numeric(n)+m,delay=numeric(n),hor=numeric(n)+H,w=1:n)
  mod <- plsr(out ~ inp, ncomp = p0, data = M)
  return (list(p=p0,m=m,mod=mod))
}
plspred<-function(Xtr,m,H,mod=NULL){

  n<-NCOL(Xtr)
  N<-NROW(Xtr)
  if (is.null(mod)){
    M=MakeEmbedded(Xtr,numeric(n)+m,delay=numeric(n),hor=numeric(n)+H,w=1:n)
    mod <- mvr(out ~ inp, ncomp = 5, data = M)
    
  }
  
  q<-NULL
  for (j in 1:n){
    for (h in seq(1:m))
      q<-c(q,Xtr[N-h+1,j])
    
  }
  
  
  yhat<-predict(mod,array(q,c(1,length(q))))
  
  return(array(yhat,c(H,n)))

}

naivepred<-function(Xtr,H){
  N<-NROW(Xtr)
  n<-NCOL(Xtr)
  return( t(array(rep(Xtr[N,],H),c(n,H))))

}
########### SSA

##


ssapred<-function(Xtr,H){
  s<-ssa(data.frame(Xtr))
  N<-NROW(Xtr)
  n<-NCOL(Xtr)
 
  return( rforecast(s, groups = list(1:6), method = "bootstrap-recurrent", len = H, R = 10))


}


########### UNI
  

unidesign<-function(Xtr,m,H,mr=FALSE){
  Kmin=5
  CC=2
  n<-NCOL(Xtr)
  N<-NROW(Xtr)
  M=MakeEmbedded(Xtr,numeric(n)+m,delay=numeric(n),hor=numeric(n)+H,w=1:n)
  fsel<-list()

  if (!mr)
    CX=cor(M$inp,M$out,use="complete.obs")
  ch=1
  for (j in 1:n){
      nm=3
      fsh=array(NA,c(H,nm))
      for (h in 1:H){
        if (!mr)
          fsh[h,]=sort(abs(CX[,ch+h-1]),decr=TRUE,index=TRUE)$ix[1:nm] ##mrmr(TS.X,TS.Y,nmax=nm)
        else
          fsh[h,]=mrmr(M$inp,M$out[,ch+h-1],nmax=nm)
      }
      ch=ch+H
      fsel[[j]]=fsh
      
    }

  return(list(Emb=M,fsel=fsel))
}

unipred<-function(Xtr,m,H,Emb,fsel){ 
  q<-NULL
  M<-Emb
  n<-NCOL(Xtr)
  N<-NROW(Xtr)
  for (j in 1:n){
    for (h in seq(1:m))
      q<-c(q,Xtr[N-h+1,j])
    
  }
  Xhat<-array(0,c(H,n))
  ch=1
  for (j in 1:n){
    
    for (h in 1:H){
      TS.Y=M$out[,ch+h-1]
      
      TS.X=M$inp[,fsel[[j]][h,]]
      
      Xhat[h,j]<-pred("lazy",TS.X,TS.Y,q[fsel[[j]][h,]],conPar=c(Kmin,CC*Kmin),linPar=NULL,classi=FALSE)
      
    }
    ch=ch+H
  }

  return(Xhat)
}



mxnetpred<-function(Xtr){
  require(mxnet)
 batch.size = 32
   seq.len = 32
   num.hidden = 16
   num.embed = 16
   num.lstm.layer = 1
   num.round = 1
   learning.rate= 0.1
   wd=0.00001
   clip_gradient=1
   update.period = 1

 model <- mx.lstm(Xtr, 
                    ctx=mx.cpu(),
                    num.round=num.round,
                    update.period=update.period,
                    num.lstm.layer=num.lstm.layer,
                    seq.len=seq.len,
                    num.hidden=num.hidden,
                    num.embed=num.embed,
                    num.label=vocab,
                    batch.size=batch.size,
                    input.size=vocab,
                    initializer=mx.init.uniform(0.1),
                    learning.rate=learning.rate,
                    wd=wd,
                    clip_gradient=clip_gradient)

 browser()
}


VARpred2<-function (x,model, h = 1, orig = 0, Out.level = F,verbose=FALSE) {
 
    Phi = model$Phi
    sig = model$Sigma
    Ph0 = model$Ph0
    p = model$order
    cnst = model$cnst
    np = dim(Phi)[2]
    k = dim(x)[2]
    nT = dim(x)[1]
    k = dim(x)[2]
    if (orig <= 0) 
        orig = nT
    if (orig > nT) 
        orig = nT
    psi = VARpsi(Phi, h)$psi
    beta = t(Phi)
    if (length(Ph0) < 1) 
        Ph0 = rep(0, k)
    if (p > orig) {
        cat("Too few data points to produce forecasts", "\n")
    }
    pred = NULL
    se = NULL
    MSE = NULL
    mse = NULL
    px = as.matrix(x[1:orig, ])
    Past = px[orig, ]
    if (p > 1) {
        for (j in 1:(p - 1)) {
            Past = c(Past, px[(orig - j), ])
        }
    }
    if (verbose)
      cat("orig ", orig, "\n")
    ne = orig - p
    xmtx = NULL
    P = NULL
    if (cnst) 
        xmtx = rep(1, ne)
    xmtx = cbind(xmtx, x[p:(orig - 1), ])
    ist = p + 1
    if (p > 1) {
        for (j in 2:p) {
            xmtx = cbind(xmtx, x[(ist - j):(orig - j), ])
        }
    }
    xmtx = as.matrix(xmtx)
    G = t(xmtx) %*% xmtx/ne
    Ginv = solve(G)
    P = Phi
    vv = Ph0
    if (p > 1) {
        II = diag(rep(1, k * (p - 1)))
        II = cbind(II, matrix(0, (p - 1) * k, k))
        P = rbind(P, II)
        vv = c(vv, rep(0, (p - 1) * k))
    }
    if (cnst) {
        c1 = c(1, rep(0, np))
        P = cbind(vv, P)
        P = rbind(c1, P)
    }
    Sig = sig
    n1 = dim(P)[2]
    MSE = (n1/orig) * sig
    for (j in 1:h) {
        tmp = Ph0 + matrix(Past, 1, np) %*% beta
        px = rbind(px, tmp)
        if (np > k) {
            Past = c(tmp, Past[1:(np - k)])
        }
        else {
            Past = tmp
        }
        if (j > 1) {
            idx = (j - 1) * k
            wk = psi[, (idx + 1):(idx + k)]
            Sig = Sig + wk %*% sig %*% t(wk)
        }
        if (j > 1) {
            for (ii in 0:(j - 1)) {
                psii = diag(rep(1, k))
                if (ii > 0) {
                  idx = ii * k
                  psii = psi[, (idx + 1):(idx + k)]
                }
                P1 = P^(j - 1 - ii) %*% Ginv
                for (jj in 0:(j - 1)) {
                  psij = diag(rep(1, k))
                  if (jj > 0) {
                    jdx = jj * k
                    psij = psi[, (jdx + 1):(jdx + k)]
                  }
                  P2 = P^(j - 1 - jj) %*% G
 k1 = sum(diag(P1 %*% P2))
                  MSE = (k1/orig) * psii %*% sig %*% t(psij)
                }
            }
        }
        se = rbind(se, sqrt(diag(Sig)))
        if (Out.level) {
            cat("Covariance matrix of forecast errors at horizon: ", 
                j, "\n")
            if (verbose){
              print(Sig)
              cat("Omega matrix at horizon: ", j, "\n")
              print(MSE)
            }
        }
        MSE = MSE + Sig
        mse = rbind(mse, sqrt(diag(MSE)))
    }
    if (verbose){
      cat("Forecasts at origin: ", orig, "\n")
      print(px[(orig + 1):(orig + h), ], digits = 4)
      cat("Standard Errors of predictions: ", "\n")
      print(se[1:h, ], digits = 4)
    }
    pred = px[(orig + 1):(orig + h), ]
    if (verbose){
      cat("Root mean square errors of predictions: ", "\n")
      print(mse[1:h, ], digits = 4)
    }
    if (orig < nT) {
      if (verbose){
        cat("Observations, predicted values,     errors, and MSE", 
            "\n")
      }
      tmp = NULL
      jend = min(nT, (orig + h))
      for (t in (orig + 1):jend) {
        case = c(t, x[t, ], px[t, ], x[t, ] - px[t, ])
        tmp = rbind(tmp, case)
      }
      colnames(tmp) <- c("time", rep("obs", k), rep("fcst", 
                                                    k), rep("err", k))
      idx = c(1)
      for (j in 1:k) {
        idx = c(idx, c(0, 1, 2) * k + j + 1)
      }
      tmp = tmp[, idx]
      if (verbose)
        print(round(tmp, 4))
    }
    VARpred <- list(pred = pred, se.err = se, mse = mse)
}
