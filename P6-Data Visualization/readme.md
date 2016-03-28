### Summary

The data set describes the effect of the rollout of a new back-end system for an online product. The new system is meant to be more stable and so its impact is measured on Support Volume, i.e. a reduction in the number of tickets that customers write in to Support is expected. Given that the user base grows while the system is rolled out, the impact is hard to measure in absolute support volume numbers, but it can be measured when looking at the ratio of tickets / users (Contact rate). This is an adaptaion of a real project I have worked on in the past (with fake but representative numbers).

![alt tag](https://raw.githubusercontent.com/bethe/da-nano/master/P6-Data%20Visualization/Data%20Vis%20Final.png)
Find the full visualization at http://bl.ocks.org/bethe/raw/2ae59fe2d4e4136339a3/

### Design
My objective was to design one data visualization that explains the impact of the new back-end system on Support volume. I decided upon a combination of Bar graphs and lines to visualize the idea. Bar graphs are great to compare absolutes and lines are great to emphasize a trend. I therefore used bar graphs for the support volume, as the most important point here was to show how the absolute numbers fluctuated. For user base and contact rate I picked a line graph, as I wanted to emphasize the growth in users and the reduction in contact rate over time. I also added dot plots to the contact rate line to make it more prominent and invite hovering over for more information.

Upon feedback of my initial draft I changed the color scheme to blue for contact rate and to light greys / white fillings for the support volume and user base growth. I also made the contact rate line stronger. This ensured the 'eyecatcher' was the downward trending contact rate line. I also changed the scales, so that both support volume and user base were starting at an index of 100. This made the graph easier to interpret. On the x-axis I converted the format to one of "Month Year" to reflect the raw data. Finally I added a 'Comment' line that could add further context in the mouse-over pop-up for each data point.



### Feedback 
Find my earliest published version at http://bl.ocks.org/bethe/raw/33c25975418d1dbbf2d4/ .  
It corresponds to my first git commit for this project from Feb 21 2016: https://github.com/bethe/da-nano/commit/79920e44fa87347bbcbf33d870f8798788e09631

I asked my girlfriend as well as two friends, both in consulting, to give me feedback upon my initial graph version. These were some of the things they said:

**(1) Too many lines in different directions, hard to tell what to focus on**  
--> In response, I picked a lighter gray for the user line and made the contact rate line stronger

**(2) Bars look too slim, overemphasizing pattern and not clear what they mean**  
--> I made them broader.

**(3) Impossible to tell the difference between the two gray shades**  
--> Made bar charts with empty filling which helped keeping them subtle and easy to distiguish in legend

**(4) X-axis is confusing / ugly in that the first data points overlap with y-axes**  
--> Fixed by added padding

**(5) X-axis labels are confusing. Why would I want to know day of the week**  
--> Fixed by translating raw data with PARSETIME

**(6) Would be helpful to get more explanations on downward trend **  
--> Added "Comment" entries to raw data and embedded via mouse-over pop-up 


### Resources
-dimple.js Documentation
-Data Visualization with d3.js course at Udacity
-Posts in Stack Overflow
