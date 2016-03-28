README.md file that includes four sections...
Summary - in no more than 4 sentences, briefly introduce your data visualization and add any context that can help readers understand it

### Summary

The data set describes the effect of the rollout of a new back-end system for an online product. The new system is meant to be more stable and so its impact is measured on Support Volume, i.e. a reduction in the number of tickets that customers write in to Support is expected. Given that the user base grows while the system is rolled out, the impact is hard to measure in absolute support volume numbers, but it can be measured when looking at the ratio of tickets / users (Contact rate). This is an adaptaion of a real project I have worked on in the past (with fake but representative numbers).

### Design
My objective was to design one data visualization that explains the impact of the new back-end system on Support volume. I decided upon a combination of Bar graphs and lines to visualize the idea. Bar graphs are great to compare absolutes and lines are great to emphasize a trend. I therefore used bar graphs for the support volume, as the most important point here was to show how the absolute numbers fluctuated. For user base and contact rate I picked a line graph, as I wanted to emphasize the growth in users and the reduction in contact rate over time. I also added dot plots to the contact rate line to make it more prominent and invite hovering over for more information.

Upon feedback of my initial draft I changed the color scheme to blue for contact rate and to light greys / white fillings for the support volume and user base growth. I also made the contact rate line stronger. This ensured the 'eyecatcher' was the downward trending contact rate line. I also changed the scales, so that both support volume and user base were starting at an index of 100. This made the graph easier to interpret. On the x-axis I converted the format to one of "Month Year" to reflect the raw data. Finally I added a 'Comment' line that could add further context in the mouse-over pop-up for each data point.


- explain any design choices you made including changes to the visualization after collecting feedback
Feedback - include all feedback you received from others on your visualization from the first sketch to the final visualization
Resources - list any sources you consulted to create your visualization
