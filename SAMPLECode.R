
##### Citation Information: Kim, H., & Lim, C. C. (2025). 
#####  Toward equitable environmental exposure modeling through convergence of data, open, and citizen sciences: 
#####   an example of air pollution exposure modeling amidst increasing wildfire smoke. 
#####  Environmental Research, 122881.

library(caret)
library(xgboost)
library(data.table)

PM25findat_SAMPLE<-read.csv("/FOLDER/SAMPLEdata.csv")
### Propensity Score Stratification Information
load(file="FOLDER/MS_INFO_PS.RData")
df_group <- PM25findat_SAMPLE[,c("MS_ID","TIMEd","TIMEh")]
df_group$order<-seq(nrow(df_group))
ms_id_group<-FIN2[!is.na(FIN2$MS_ID),c("MS_ID","group")]
df_group<-merge(df_group,ms_id_group,by="MS_ID")
df_group<-df_group[order(df_group$order),]
table(FIN2$group,FIN2$MONITOR)

tttt<-table(FIN2$group,FIN2$MONITOR)
p.w<-tttt[,1]/min(tttt[,1]) ### Weight
p.w<-data.frame(group=seq(0,(nrow(tttt)-1)),p.w=p.w)

FIN2.MSINFO<-subset(FIN2,MONITOR==1)
FIN2.MSINFO<-merge(FIN2.MSINFO,p.w,by="group")
spatialevalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  group_ind<-c(df_group[TUNSELECTED,"group"])  ## IF YOU USE the package "data.table", you might need to add "$group" here
  group_ind_labels<-aggregate(labels~group_ind,FUN=mean)
  group_ind_preds<-aggregate(preds~group_ind,FUN=mean)
  sp.e<-merge(group_ind_labels,group_ind_preds,by="group_ind")
  spatial.err<-mean((sp.e[,2]-sp.e[,3])^2)
  u<-(preds-labels)^2
  overal.err<-mean(u)
  err <- sqrt((overal.err)*(1-alpha_val)+(spatial.err)*alpha_val)
  return(list(metric = "EqualityPenalty", value = err))
}
spatialobjective <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  group<-c(df_group[SELECTED,"group"])  ## IF YOU USE the package "data.table", you might need to add "$group" here
  group_labels<-aggregate(labels~group,FUN=mean)
  group_preds<-aggregate(preds~group,FUN=mean)
  df_group2<-df_group[SELECTED,]
  df_group2<-merge(df_group2,group_labels,by="group")
  df_group2<-merge(df_group2,group_preds,by="group")
  df_group2<-df_group2[order(df_group2$order),]
  grad<- (1-alpha_val)*(preds-labels)+(alpha_val)*(df_group2$preds-df_group2$labels)
  hess<- rep(1, length(preds))
  return(list(grad = grad, hess = hess))
}
mse<-function(x,y) { mean((x-y)^2)}

eval_error_met<-spatialevalerror  ## WE WILL USE THIS FOR XGBOOST's EVALUATION  METRIC
objective_fun<-spatialobjective   ## WE WILL USE THIS FOR XGBOOST's OBJECTIVE FUNCTION


nrun<-5  ## HOW MANY MODELS DO YOU USE?
alpha_val<-c(0.9,0.5,0.25,0.1,0)  ## ALPHA VALUE FOR THE NEW LOSS-FUNCTION
alpha_val<-alpha_val[1]  

########
y <- PM25findat_SAMPLE$PM25  ## YOUR LABEL (DEPENDENT VARIABLE)
df <- within(PM25findat_SAMPLE, rm(PM25,TIMEd,TIMEh,Date2,Date,TIME_ID,weekdays,HR_UTC))   ### DATA FRAME WITH PREDICTORS
df$order<-seq(nrow(df)) 

df<-data.frame(df)  
df<-merge(df,FIN2.MSINFO[,c("MS_ID","p.w")],by="MS_ID")
df<-df[order(df$order),]
df.p.w<-df[,c("p.w")]   ### weights, related to PS.
df <- within(df, rm(order))   ### DATA FRAME WITH PREDICTORS

'%!in%' <- function(x,y)!('%in%'(x,y))

##Lists to store datasets: WE NEED LISTS BECAUSE WE WILL HAVE MULTIPLE MODELS based on MULTIPLE SAMPLES from PS.
df_train_set<-vector("list",nrun)
y_train_set<-vector("list",nrun)
df_test_set<-vector("list",nrun)
y_test_set<-vector("list",nrun)
df_tuning_set<-vector("list",nrun)
y_tuning_set<-vector("list",nrun)
df_Cal_set<-vector("list",nrun)
y_Cal_set<-vector("list",nrun)

df_train_set2<-vector("list",nrun)  ## train set after the automatic variable selection
df_test_set2<-vector("list",nrun)  ## updated test set reflecting the automatic variable selection (identical with df_test_set except for the number of columns)
df_Cal_set2<-vector("list",nrun) ## updated calibration set reflecting the automatic variable selection (identical with df_test_set except for the number of columns)
nn_test_ind<-vector("list",nrun) ## to save the index of the observations selected for the test set
nn_tuning_ind<-vector("list",nrun)  ## to save the index of the observations selected for the training set
nn_Cal_ind<-vector("list",nrun) ## to save the index of the observations selected for the calibration set
F_Sel<-F_Sel2<-vector("list",nrun) ## to save the predictors selected for model training

group_list<-length(unique(FIN2.MSINFO$group)) ## how many # of PS groups?

moni_Sel<-vector("list",group_list)  ## to save the monitors/sensors used for training set
moni_Val<-vector("list",nrun)  ## to save the monitors/sensors used for test set
moni_Tun<-vector("list",nrun)  ## to save the monitors/sensors used for tuning set
moni_Ct<-vector("list",group_list)

for (nn in seq(nrun)) {
  moni_Val[[nn]]<-moni_Tun[[nn]]<-vector("list",group_list)
}

Valdat_set<-vector("list",nrun)  ## to save the test set with predicted Ys

fit2<-vector("list",nrun) ## to save the final XGBoost models
tp.correct<-sp.correct<-vector("list",nrun) ## to save the calibration models (TemPoral calibration; SPatial calibration)

'%!in%' <- function(x,y)!('%in%'(x,y))
set.seed(42)
for (nn in seq(nrun)) {
  set.seed(nn*1842423)
  start.time<-Sys.time()
  for (ii in seq(group_list)) {
    moni<-FIN2.MSINFO[FIN2.MSINFO$group==ii-1,"MS_ID"]
    moni<-moni[sample(length(moni))]
    
    if(length(moni)==5) {
      selmo<-moni[1:3]
      ctmo<- moni[2]
      tunmo<- moni[2:3]
      valmo<- moni[4:5]
    }
    
    if(length(moni)==6) {
      selmo<-moni[c(1:3)]
      ctmo<- moni[3]
      tunmo<- moni[3:4]
      valmo<- moni[5:6]
    }
    
    if(length(moni)==7) {
      selmo<-moni[c(1:3)]
      ctmo<- moni[4]
      tunmo<- moni[4:5]
      valmo<- moni[6:7]
    }
    
    if(length(moni)==9) {
      selmo<-moni[c(1:3)]
      ctmo<- moni[4]
      tunmo<- moni[5:6]
      valmo<- moni[7:8]
    }
    
    if(length(moni)==12) {
      selmo<-moni[c(1:3)]
      ctmo<- moni[4]
      tunmo<- moni[5:6]
      valmo<- moni[7:8]
    }
    
    if(length(moni)==18) {
      selmo<-moni[c(1:3)]
      ctmo<- moni[4]
      tunmo<- moni[5:6]
      valmo<- moni[7:8]
    }
    
    if(length(moni)==20) {
      selmo<-moni[c(1:3)]
      ctmo<- moni[4]
      tunmo<- moni[5:6]
      valmo<- moni[7:8]
    }
    if(length(moni)==34) {
      selmo<-moni[c(1:3)]
      ctmo<- moni[4]
      tunmo<- moni[5:6]
      valmo<- moni[7:8]
    }
    
    moni_Sel[[ii]]<-selmo
    moni_Val[[nn]][[ii]]<-valmo
    moni_Ct[[ii]]<-ctmo
    moni_Tun[[nn]][[ii]]<-tunmo
    
  }
  
  moni_Sel2<-unlist(moni_Sel)
  moni_Ct2<-unlist(moni_Ct)
  moni_Val2<-unlist(moni_Val[[nn]])
  moni_Tun2<-unlist(moni_Tun[[nn]])
  
  SELECTED<-which(PM25findat_SAMPLE$MS_ID %in% moni_Sel2)
  VALSELECTED<-which(PM25findat_SAMPLE$MS_ID %in% moni_Val2)
  TUNSELECTED<-which(PM25findat_SAMPLE$MS_ID %in% moni_Tun2)
  CTSELECTED<-which(PM25findat_SAMPLE$MS_ID %in% moni_Ct2)
  
  weighting_train<-df.p.w[SELECTED]
  weighting_cal<-df.p.w[CTSELECTED]
  weighting_tuning<-df.p.w[TUNSELECTED]
  weighting_val<-df.p.w[VALSELECTED]
  
  df_train_set[[nn]]<-df[SELECTED, ]
  y_train_set[[nn]]<-y[SELECTED]
  
  df_test_set[[nn]]<-df[VALSELECTED, ]
  y_test_set[[nn]]<-y[VALSELECTED]
  nn_test_ind[[nn]]<-VALSELECTED
  
  df_tuning_set[[nn]]<-df[TUNSELECTED, ]
  y_tuning_set[[nn]]<-y[TUNSELECTED]
  nn_tuning_ind[[nn]]<-TUNSELECTED
  
  df_Cal_set[[nn]]<-df[CTSELECTED, ]
  y_Cal_set[[nn]]<-y[CTSELECTED]
  nn_Cal_ind[[nn]]<-CTSELECTED
  
  predictor<-as.matrix(df_train_set[[nn]])
  dependent<-as.matrix(y_train_set[[nn]])
  INPUTMAT_CV <- xgb.DMatrix(data = predictor, label = dependent,weight=weighting_train)
  
  predictor<-as.matrix(df_tuning_set[[nn]])
  dependent<-as.matrix(y_tuning_set[[nn]])
  TUNMAT_CV <- xgb.DMatrix(data = predictor, label = dependent,weight=weighting_tuning)
  
  
  watchlist <- list(eval = TUNMAT_CV)
  
  fit1<-xgb.train(params=list(
    eval_metric=eval_error_met,
    objective = objective_fun,
    max.depth = 5,eta=0.3,subsample=0.1,colsample_bytree=0.1,gamma=0),
    data = INPUTMAT_CV, nrounds = 10000,nthread=-1,verbose=0,watchlist,maximize=FALSE,early_stopping_rounds=10,seed = 1423242523323)
  
  evalmm1<-attr(fit1,"evaluation_log")$eval_EqualityPenalty
  niter1<-which(evalmm1==min(evalmm1))
  
  Importance.Predictors<-xgb.importance(colnames(INPUTMAT_CV), model = fit1)
  Features_Sel<-c(Importance.Predictors$Feature[which(Importance.Predictors$Gain>0.001)],
                  Importance.Predictors$Feature[which(Importance.Predictors$Frequency>0.0015)])
  Features_Sel<-Features_Sel[!duplicated(Features_Sel)]
  
  df_train_set2[[nn]]<-df[SELECTED,Features_Sel]
  df_test_set2[[nn]]<-df[VALSELECTED,Features_Sel]
  df_Cal_set2[[nn]]<-df[CTSELECTED,Features_Sel]
  
  predictor<-as.matrix(df_train_set2[[nn]])
  dependent<-as.matrix(y_train_set[[nn]])
  INPUTMAT_CV3 <- xgb.DMatrix(data = predictor, label = dependent,weight=weighting_train)
  
  predictor<-as.matrix(df_tuning_set[[nn]][,Features_Sel])
  dependent<-as.matrix(y_tuning_set[[nn]])
  TUNMAT_CV2 <- xgb.DMatrix(data = predictor, label = dependent,weight=weighting_tuning)
  watchlist <- list(eval = TUNMAT_CV2)
  
  rm(fit1)
  fit2[[nn]]<-xgb.train(params=list(
    eval_metric=eval_error_met,
    objective = objective_fun,
    max.depth = round(runif(1,5,15)),eta=0.1,subsample=runif(1,0.08,0.12),colsample_bytree=runif(1,0.08,0.12),gamma=0),
    data = INPUTMAT_CV3, nrounds = 10000,nthread=-1,verbose=0,watchlist,maximize=FALSE,early_stopping_rounds=10,seed = 1423242523323)
  
  evalmm2<-attr(fit2[[nn]],"evaluation_log")$eval_EqualityPenalty
  niter2<-which(evalmm2==min(evalmm2))
  
  F_Sel[[nn]]<-c(niter1,min(evalmm1),niter2,min(evalmm2))
  
  Cal_dat<-PM25findat_SAMPLE[nn_Cal_ind[[nn]], ]
  predictor<-as.matrix(df_Cal_set2[[nn]])
  dependent<-as.matrix(y_Cal_set[[nn]])
  CALMAT <- xgb.DMatrix(data = predictor, label = dependent,weight=weighting_cal)
  
  Cal_dat<-data.frame(MS_ID=Cal_dat$MS_ID,
                      TIMEh=Cal_dat$TIMEh,
                      TIMEd=Cal_dat$TIMEd,
                      Date=Cal_dat$Date,
                      weekdays=Cal_dat$weekdays,
                      PM25=y_Cal_set[[nn]],
                      PM25_pred=predict(fit2[[nn]],newdata=CALMAT),
                      weighting_cal=weighting_cal)
  
  ###SPACE & TIME CALIBRATION
  PM25_pred_sp<-aggregate(PM25_pred~MS_ID,FUN=mean,data=Cal_dat)
  colnames(PM25_pred_sp)[2]<-"PM25_pred_sp"
  
  PM25_sp<-aggregate(PM25~MS_ID,FUN=mean,data=Cal_dat)
  colnames(PM25_sp)[2]<-"PM25_sp"
  
  sp.dat<-merge(PM25_pred_sp,PM25_sp,by="MS_ID")
  sp.correct[[nn]]<-lm(PM25_sp~PM25_pred_sp,data=sp.dat)  ###SPATIAL CALIBRATION MODEL
  
  Cal_dat<-merge(Cal_dat,PM25_pred_sp,by="MS_ID")
  Cal_dat<-merge(Cal_dat,PM25_sp,by="MS_ID")
  
  Cal_dat$PM25_tp<-Cal_dat$PM25-Cal_dat$PM25_sp
  Cal_dat$PM25_pred_tp<-Cal_dat$PM25_pred-Cal_dat$PM25_pred_sp
  tp.correct[[nn]]<-lm(PM25_tp~PM25_pred_tp,data=Cal_dat)   ###TEMPORAL CALIBRATION MODEL
  
  testdat<-PM25findat_SAMPLE[nn_test_ind[[nn]], ]
  predictor<-as.matrix(df_test_set2[[nn]])
  dependent<-as.matrix(y_test_set[[nn]])
  VALMAT <- xgb.DMatrix(data = predictor, label = dependent,weight=weighting_val)
  
  Valdat_set[[nn]]<-data.frame(IND=seq(nrow(testdat)),MS_ID=testdat$MS_ID,TIMEh=testdat$TIMEh,TIMEd=testdat$TIMEd,Date=testdat$Date,
                               weekdays=testdat$weekdays,
                               PM25=y_test_set[[nn]],
                               PM25_pred=predict(fit2[[nn]],newdata=VALMAT),
                               weighting_val=weighting_val)
  
  ###SPACE & TIME CALIBRATION
  v.sp<-aggregate(PM25_pred~MS_ID,data=Valdat_set[[nn]],FUN=mean)
  colnames(v.sp)[2]<-c("PM25_pred_sp")
  
  Valdat_set[[nn]]<-merge(Valdat_set[[nn]],v.sp,by="MS_ID")
  
  Valdat_set[[nn]]$PM25_pred_tp<-Valdat_set[[nn]]$PM25_pred-Valdat_set[[nn]]$PM25_pred_sp
  Valdat_set[[nn]]$PM25_pred2_tp<-predict(tp.correct[[nn]],newdata=Valdat_set[[nn]])
  Valdat_set[[nn]]$PM25_pred2<-Valdat_set[[nn]]$PM25_pred2_tp+Valdat_set[[nn]]$PM25_pred_sp #Time-corrected
  
  Valdat_set[[nn]]$PM25_pred2_sp<-predict(sp.correct[[nn]],newdata=Valdat_set[[nn]])
  Valdat_set[[nn]]$PM25_pred3<-Valdat_set[[nn]]$PM25_pred2_sp+Valdat_set[[nn]]$PM25_pred_tp #Space-corrected
  
  Valdat_set[[nn]]$PM25_pred4<-Valdat_set[[nn]]$PM25_pred2_tp+Valdat_set[[nn]]$PM25_pred2_sp #both corrected
  
  end.time<-Sys.time()
  elapsed<-end.time-start.time
  print("Valdat")
  print(paste(nn,round(elapsed,2),attr(elapsed,"units")))
  print(F_Sel[[nn]])
}

### TRY THIS:
par(mfrow=c(2,2)) ## IF YOU WONDER WHY THESE COULD BE SIMILAR TO ONE ANOTHER, CHECK CALIBRATION MODELS
plot(Valdat_set[[nn]]$PM25_pred,Valdat_set[[nn]]$PM25)
plot(Valdat_set[[nn]]$PM25_pred2,Valdat_set[[nn]]$PM25)
plot(Valdat_set[[nn]]$PM25_pred3,Valdat_set[[nn]]$PM25)
plot(Valdat_set[[nn]]$PM25_pred4,Valdat_set[[nn]]$PM25)





### FINAL TEST SET TO SEE VALIDATION METRICS  (HOURLY PM2.5)

L1_Valdat<-do.call(rbind,Valdat_set)  
qqq<-aggregate(PM25~MS_ID+TIMEh,data=L1_Valdat,FUN=mean)  
www<-aggregate(PM25_pred4~MS_ID+TIMEh,data=L1_Valdat,FUN=mean) ## WE WILL EXAMINE "PRED4"; ## MODEL AVERAGING
Valdat<-merge(qqq,www,by=c("MS_ID","TIMEh"))
PM25_spatial<-aggregate(PM25~MS_ID,data=Valdat,FUN=mean)
colnames(PM25_spatial)[2]<-"PM25_sp"
Valdat<-merge(Valdat,PM25_spatial,by="MS_ID")
Valdat$PM25_tp<-Valdat$PM25-Valdat$PM25_sp

PM25_spatial<-aggregate(PM25_pred4~MS_ID,data=Valdat,FUN=mean) ## WE WILL EXAMINE "PRED4"
colnames(PM25_spatial)[2]<-"PM25_pred4_sp"
Valdat<-merge(Valdat,PM25_spatial,by="MS_ID")
Valdat$PM25_pred4_tp<-Valdat$PM25_pred4-Valdat$PM25_pred4_sp
Valdat<-merge(Valdat,FIN2.MSINFO[,c("MS_ID","p.w")],by="MS_ID")

library(tibble)
library(ggpointdensity)
library(ggplot2)
library(viridis)
specify_decimal <- function(x, k) trimws(format(round(x, k), nsmall=k))

VALfit<-lm(PM25~PM25_pred4,data=Valdat,weights=p.w)
Valdat$XGB1pred_c_Val_pred_cline<-predict(VALfit,newdata=Valdat)
Valfit.coef<-specify_decimal(c(
  summary(VALfit)$coef[2,1],
  summary(VALfit)$coef[2,1]-1.96*summary(VALfit)$coef[2,2],
  summary(VALfit)$coef[2,1]+1.96*summary(VALfit)$coef[2,2]
),3)

Valfit.int.coef<-specify_decimal(c(
  summary(VALfit)$coef[1,1],
  summary(VALfit)$coef[1,1]-1.96*summary(VALfit)$coef[1,2],
  summary(VALfit)$coef[1,1]+1.96*summary(VALfit)$coef[1,2]
),3)


Valfit.coef.R.sq<-paste0(specify_decimal(summary(VALfit)$r.sq*100,1),"%")
dat <- 
  tibble(x = Valdat[,"PM25_pred4"],
         y = Valdat[,"PM25"])
Plot.Val.PM25<-ggplot(data = dat, mapping = aes(x = x, y = y)) +
  geom_pointdensity() +
  scale_color_viridis()+theme_bw()+theme(legend.title = element_blank())+
  ylab(expression("PM"[2.5]~"Measured"~mu*g/m^3))+
  xlab(expression("PM"[2.5]~"Estimated"~mu*g/m^3))+
  guides(fill=guide_legend(title=""))+
  annotate(geom='text', label=paste0("y = ",
                                     Valfit.int.coef[1]," (95% CI: " ,Valfit.int.coef[2],", ",
                                     Valfit.int.coef[3],") + \n",
                                     
                                     Valfit.coef[1]," (95% CI: " ,Valfit.coef[2],", ",
                                     Valfit.coef[3],") * x \n R-sq: ",Valfit.coef.R.sq),
           x=Inf,y=-Inf,hjust=1,vjust=-0.5)+
  geom_abline(slope=1,intercept=0,lty=2,col="grey")+
  geom_abline(slope=1,intercept=0,lty=2,col="grey")+
  geom_line(data=Valdat,aes(x=PM25_pred4,y=XGB1pred_c_Val_pred_cline),col="red",lty=2)+
  ggtitle("A. Hourly (Overall)")





VALfit.temporal<-lm(PM25_tp~PM25_pred4_tp,data=Valdat,weights=p.w)
Valdat$XGB1pred_c_Val_Temporal_pred_cline<-predict(VALfit.temporal,newdata=Valdat)
Valfit.temporal.coef<-specify_decimal(c(
  summary(VALfit.temporal)$coef[2,1],
  summary(VALfit.temporal)$coef[2,1]-1.96*summary(VALfit.temporal)$coef[2,2],
  summary(VALfit.temporal)$coef[2,1]+1.96*summary(VALfit.temporal)$coef[2,2]
),3)
Valfit.temporal.int.coef<-specify_decimal(c(
  summary(VALfit.temporal)$coef[1,1],
  summary(VALfit.temporal)$coef[1,1]-1.96*summary(VALfit.temporal)$coef[1,2],
  summary(VALfit.temporal)$coef[1,1]+1.96*summary(VALfit.temporal)$coef[1,2]
),3)


Valfit.temporal.coef.R.sq<-paste0(specify_decimal(summary(VALfit.temporal)$r.sq*100,1),"%")
dat <- 
  tibble(x = Valdat[,"PM25_pred4_tp"],
         y = Valdat[,"PM25_tp"])
Plot.Val.PM25.Temporal<-ggplot(data = dat, mapping = aes(x = x, y = y)) +
  geom_pointdensity() +
  scale_color_viridis()+theme_bw()+theme(legend.title = element_blank())+
  ylab(expression("PM"[2.5]~"Measured - Spatial mean"~mu*g/m^3))+
  xlab(expression("PM"[2.5]~"Estimated - Spatial mean"~mu*g/m^3))+
  guides(fill=guide_legend(title=""))+
  annotate(geom='text', label=paste0("y = ",
                                     Valfit.temporal.int.coef[1]," (95% CI: " ,Valfit.temporal.int.coef[2],", ",
                                     Valfit.temporal.int.coef[3],") + \n",
                                     
                                     Valfit.temporal.coef[1]," (95% CI: " ,Valfit.temporal.coef[2],", ",
                                     Valfit.temporal.coef[3],") * x \n R-sq: ",Valfit.temporal.coef.R.sq),
           x=Inf,y=-Inf,hjust=1,vjust=-0.5)+
  geom_abline(slope=1,intercept=0,lty=2,col="grey")+
  geom_abline(slope=1,intercept=0,lty=2,col="grey")+
  geom_line(data=Valdat,aes(x=PM25_pred4_tp,y=XGB1pred_c_Val_Temporal_pred_cline),col="red",lty=2)+
  ggtitle("B. Hourly (Temporal)")





Valdat_sp<-Valdat[!duplicated(Valdat$MS_ID),]
VALfit.spatial<-lm(PM25_sp~PM25_pred4_sp,data=Valdat_sp,weights=p.w)
Valdat_sp$XGB1pred_c_Val_spatial_pred_cline<-predict(VALfit.spatial,newdata=Valdat_sp)
Valfit.spatial.coef<-specify_decimal(c(
  summary(VALfit.spatial)$coef[2,1],
  summary(VALfit.spatial)$coef[2,1]-1.96*summary(VALfit.spatial)$coef[2,2],
  summary(VALfit.spatial)$coef[2,1]+1.96*summary(VALfit.spatial)$coef[2,2]
),3)
Valfit.spatial.int.coef<-specify_decimal(c(
  summary(VALfit.spatial)$coef[1,1],
  summary(VALfit.spatial)$coef[1,1]-1.96*summary(VALfit.spatial)$coef[1,2],
  summary(VALfit.spatial)$coef[1,1]+1.96*summary(VALfit.spatial)$coef[1,2]
),3)
Valfit.spatial.coef.R.sq<-paste0(specify_decimal(summary(VALfit.spatial)$r.sq*100,1),"%")
dat <- 
  tibble(x = Valdat_sp[,"PM25_pred4_sp"],
         y = Valdat_sp[,"PM25_sp"])
Plot.Val.PM25.Spatial<-ggplot(data = dat, mapping = aes(x = x, y = y)) +
  geom_point() +
  scale_color_viridis()+theme_bw()+theme(legend.title = element_blank())+
  ylab(expression("PM"[2.5]~"Measured"~mu*g/m^3))+
  xlab(expression("PM"[2.5]~"Estimated"~mu*g/m^3))+
  guides(fill=guide_legend(title=""))+
  annotate(geom='text', label=paste0("y = ",
                                     Valfit.spatial.int.coef[1]," (95% CI: " ,Valfit.spatial.int.coef[2],", ",
                                     Valfit.spatial.int.coef[3],") + \n",
                                     
                                     Valfit.spatial.coef[1]," (95% CI: " ,Valfit.spatial.coef[2],", ",
                                     Valfit.spatial.coef[3],") * x \n R-sq: ",Valfit.spatial.coef.R.sq),
           x=Inf,y=-Inf,hjust=1,vjust=-0.5)+
  geom_abline(slope=1,intercept=0,lty=2,col="grey")+
  geom_abline(slope=1,intercept=0,lty=2,col="grey")+
  geom_line(data=Valdat_sp,aes(x=PM25_pred4_sp,y=XGB1pred_c_Val_spatial_pred_cline),col="red",lty=2)+
  ggtitle("C. Hourly (Spatial)")


Plot.Val.PM25_h<-Plot.Val.PM25
Plot.Val.PM25.Temporal_h<-Plot.Val.PM25.Temporal
Plot.Val.PM25.Spatial_h<-Plot.Val.PM25.Spatial

library(gridExtra)
library(grid)
library(ggplot2)
library(lattice)
grid.arrange(Plot.Val.PM25_h,Plot.Val.PM25.Temporal_h,Plot.Val.PM25.Spatial_h, ncol=3) 
### NOTE THAT MODELS THAT WILL BE TRAINED USING "SAMPLEData.csv" will not be good, 
### BECAUSE The SAMPLEData.csv includes only some observations used for this study and does not include many informative variables used for this study. 


