if (!require("gplots")) {
  install.packages("gplots", dependencies = TRUE)
  library(gplots)
}
if (!require("RColorBrewer")) {
  install.packages("RColorBrewer", dependencies = TRUE)
  library(RColorBrewer)
}

library(ggplot2)
library(reshape2)

top_number <- 20

df_res_all <- read.csv("L3_coefs_proc.csv",sep="\t",header=F)

df_res_all_mat <- acast(df_res_all, V2~V1, value.var="V3")
df_res_all_mat[is.na(df_res_all_mat)] <- 0.0
df_res_all_mat <- df_res_all_mat/apply(df_res_all_mat,1,max)
top_sel <- names(sort(rowSums(df_res_all_mat),decreasing=T)[1:top_number])

my_palette <- colorRampPalette(c("white","yellow","orange","darkorange","red","darkred","black"))(n = 250)

heatmap.2(df_res_all_mat[top_sel,],density.info="none",trace="none",col=my_palette)


df_res_all <- read.csv("L3_coefs_proc_nodup.csv",sep="\t",header=F)

df_res_all_mat <- acast(df_res_all, V2~V1, value.var="V3")
df_res_all_mat[is.na(df_res_all_mat)] <- 0.0
df_res_all_mat <- df_res_all_mat/apply(df_res_all_mat,1,max)
top_sel <- names(sort(rowSums(df_res_all_mat),decreasing=T)[1:top_number])

my_palette <- colorRampPalette(c("white","yellow","orange","darkorange","red","darkred","black"))(n = 250)

heatmap.2(df_res_all_mat[top_sel,],density.info="none",trace="none",col=my_palette)



