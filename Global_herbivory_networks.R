## Code for Using ecological networks to answer questions in global biogeography and ecology ## 

# CODE COLLATED AND WRITTEN BY F.M. WINDSOR 
# FOR ENQUIRIES PLEASE CONTACT fredric.windsor@newcastle.ac.uk


######################################################################################
##################################### SET-UP #########################################
######################################################################################

# Clear the environment and change the directory
rm(list=ls())
setwd("~/data")

# Attach all relevant libraries 
library(maps); library(ggplot2); library(ggsci); library(ggimage); library(gridExtra); 
library(tidyverse); library(grid); library(lme4); library(LMERConvenienceFunctions);
library(lmerTest); library(magrittr)


######################################################################################
################################### DATA INPUT #######################################
######################################################################################

# Read in the collated network data
dframe1 <- read.csv("") # Supplementary data table 1
dframe2 <- read.csv("") # Supplementary data table 2
dframe3 <- left_join(dframe1, dframe2)

######################################################################################
###################################### FIGURES #######################################
######################################################################################

#### Figure 1 ####

# a. Map the distribution of the records 
world <- map_data("world")
world <- subset(world, region != "Antarctica")

plot1 <- ggplot() + 
  geom_polygon(aes(long, lat, group=group), data = world, size = 0.25, colour = "black", fill = "grey80") +  
  theme(axis.text = element_blank(), axis.line = element_blank(), axis.ticks = element_blank(), 
        panel.border = element_blank(), panel.grid = element_blank(),
        axis.title = element_blank(), plot.margin = unit(c(-5,-8,-7.5,-8), "mm"), 
        panel.background = element_blank(), legend.position = c(0.11,0.45), 
        legend.background = element_rect(fill = "white"), 
        legend.key = element_blank(), legend.direction = "vertical") + 
   geom_point(aes(longitude, latitude),fill = "darkgreen", size = 5, colour = "black", pch = 21, data = dframe1, na.rm = T) +
   scale_fill_d3(name = "Network type")
plot1 # 939 x 455

best_models <- select(dframe3, connectance, Tree_Density, GlobBiomass_AboveGroundBiomass, 
                      total_interactions, GlobBiomass_GrowingStockVolume, SG_Clay_Content_030cm,
                      SG_Clay_Content_015cm, SG_Clay_Content_200cm, SG_Soil_pH_H2O_000cm)

plot1 <- ggplot(aes(x=SG_Soil_pH_H2O_000cm/10, y=connectance), data = best_models) + 
  stat_summary(fill = "darkgreen", pch = 21, colour = "black", geom = "point", fun = mean, size = 5) + 
  geom_smooth(method = "lm", colour = "black", size = 2) + 
  theme_bw() + 
  theme(axis.title = element_text(size = 14),
        axis.text = element_text(size = 12, colour = "black")) + 
  ylab("") + 
  annotate(geom = "text", label = "R^2 == 0.25", x = 7.5, y = 0.35, parse = T, size = 5) + 
  xlab("Soil pH")
plot1

plot2 <- ggplot(aes(x=GlobBiomass_GrowingStockVolume, y=connectance), data = best_models[-174,]) + 
  stat_summary(fill = "darkgreen", pch = 21, colour = "black", geom = "point", fun.y = mean, size = 5) + 
  stat_smooth(method="lm", se=TRUE, formula=y ~ poly(x, 2, raw=TRUE),colour="black",size = 2) +  
  theme_bw() + 
  theme(axis.title = element_text(size = 14), axis.text = element_text(size = 12, colour = "black")) + 
  ylab(expression(paste("Connectance (l/",~s^2,")"))) + 
  xlab(expression(paste("Growing Stock Volume (",m^3~ha^-1,")"))) + 
  annotate(geom = "text", label = "R^2 == 0.29", x = 275, y = 0.35, parse = T, size = 5)
plot2
