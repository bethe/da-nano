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
  r <- c(min(x), mean(x) - 0.6745*sd(x), mean(x), mean(x) + 0.6745*sd(x), max(x))
  names(r) <- c("ymin", "lower", "middle", "upper", "ymax")
  r
}

# ggplot code
ggplot(aes(y=Seconds, x=factor(Treatment), fill=Treatment), data = stroop) +
  stat_summary(fun.data = min.mean.sd.max, geom = "boxplot") + 
  geom_jitter(position=position_jitter(width=.2), size=3) + 
  ggtitle(expression(atop("Time it took people to read by treatment group",
    atop(italic("Boxplot with means, quantiles, and range"))))) +
  xlab("Treatment Groups") + 
  ylab("Time (Seconds)")


## 5 Independent samples t-test
t.test(raw$Congruent, raw$Incongruent, alternative = "less", paired = TRUE)
