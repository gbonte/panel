
## TEST: predict(encoder2,Xts,hidden.output=FALSE)$X.out-decode(encoder2,predict(encoder2,Xts,hidden.output=TRUE)$X.out)
 

activation <- function(z,unit.type) {
        if (unit.type == "logistic") 
            return(1/(1 + exp(-z)))
        if (unit.type == "tanh") 

          return(tanh(z))
    }

 decode <- function(object, Z) {
   rescale.back <- function(X.in, X.in.min, X.in.max, unit.type, 
        offset) {
        if (unit.type == "logistic") {
            X.in <- (X.in - offset)/(1 - 2 * offset) * (X.in.max - 
                X.in.min) + X.in.min
        }
        if (unit.type == "tanh") {
            X.in <- (X.in - offset + 1)/(1 - offset) * (X.in.max - 
                X.in.min)/2 + X.in.min
        }
        return(list(X.rescaled = X.in))
    }
   if (class(object) == "autoencoder") {
     W <- object$W
     b <- object$b
     nl <- object$nl
     sl <- object$sl
     N.hidden <- object$N.hidden
     unit.type <- object$unit.type
     rescale.flag <- object$rescaling$rescale.flag
     rescaling.min <- object$rescaling$rescaling.min
     rescaling.max <- object$rescaling$rescaling.max
     rescaling.offset <- object$rescaling$rescaling.offset
   }
   
   NrowX = NROW(Z)
   z <- list()
   a <- list()
   a[[2]] <- (Z)
   for (l in 3:nl) {
     z[[l]] <- t(W[[l - 1]] %*% t(a[[l - 1]])) + matrix(b[[l - 
                                                           1]], nrow = NrowX, ncol = length(b[[l - 1]]), 
                                                        byrow = TRUE)
     a[[l]] <- activation(z[[l]],unit.type)
   }
   output=a[[nl]]
   if (rescale.flag) {
      output <- rescale.back(X.in = output, X.in.min = rescaling.min, 
                X.in.max = rescaling.max, unit.type = unit.type, 
                offset = rescaling.offset)$X.rescaled
        }
   return(output)
 }


