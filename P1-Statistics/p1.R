## Project 1 for Udacity Data Analyst Nanodegree

library(ggplot2)

## 1 Import Data
raw = read.csv("stroopdata.csv")

## 2 Transform Data Format
stroop <- raw$Congruent
stroop <- append(stroop, raw$Incongruent)
stroop <- as.data.frame(stroop)
stroop$Treatment <- c(rep("Congruent", 24), rep("Incongruent", 24))
colnames(stroop) <-c("Seconds", "Treatment" )


## 3 Plot Histograms & Density in parallel
ggplot(data = stroop, aes(x=Seconds, y=..density.., fill=Treatment)) +
  geom_histogram(binwidth = 1, alpha = 0.5) +
  geom_density(alpha = 0.5) +
  ggtitle(expression(atop("Time it took people to read by treatment group",
                    atop(italic("Histograms and Curve Estimates")))))


## 4 Plot Boxplots to compare means & standard deviations
# function for computing mean, DS, max and min values
min.mean.sd.max <- function(x) {
  r <- c(min(x), mean(x) - 1.96*sd(x), mean(x), mean(x) + 1.96*sd(x), max(x))
  names(r) <- c("ymin", "lower", "middle", "upper", "ymax")
  r
}

# ggplot code
ggplot(aes(y=Seconds, x=factor(Treatment), fill=Treatment), data = stroop) +
  stat_summary(fun.data = min.mean.sd.max, geom = "boxplot") + 
  geom_jitter(position=position_jitter(width=.2), size=3) + 
  ggtitle(expression(atop("Time it took people to read by treatment group",
    atop(italic("Boxplot with means, 95%CI, and maxima by group"))))) +
  xlab("Treatment Groups") + 
  ylab("Time (Seconds")



## 2 Parallel Histograms
ggplot(data = raw) +
  geom_histogram(aes(x=Congruent), binwidth = 2, fill = "blue", alpha = 0.7) +
  geom_histogram(aes(x=Incongruent), binwidth = 2, fill = "red", alpha = 0.7)

## 3 Histograms with Density curve
ggplot(data = raw) +
  geom_histogram(aes(x=Congruent, y = ..density..), binwidth = 2, fill = "blue", alpha = 0.4) +
  geom_histogram(aes(x=Incongruent, y = ..density..), binwidth = 2, fill = "red", alpha = 0.4) +
  geom_density(aes(x=Congruent), fill = "blue", alpha = 0.5) +
  geom_density(aes(x=Incongruent), fill = "red", alpha = 0.5) +
  xlab("Time to read (s)") +
  labs("color = colnames")

## 3 Histograms next to each other
ggplot(data = raw) +
  geom_density(aes(x=Congruent))
  
  geom_histogram(data = raw, aes(x=Congruent), binwidth = 2, fill = "blue", alpha = 0.7) +
  geom_histogram(data = raw, aes(x=Incongruent), binwidth = 2, fill = "red", alpha = 0.7) +
  
