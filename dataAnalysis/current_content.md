For this classification task, we analyzed a dataset of survey responses related to three food items: Pizza, Shawarma, and Sushi. The survey included eight questions covering various aspects of food perception, from complexity and cost to associated settings and drink pairings. Our exploration focused on understanding how these features distribute across the three food types to identify potential discriminative patterns for classification. The analysis revealed distinct distribution patterns across food types for most features, which is useful for the feature selection performed later.
Section 1: data exploration for each question.

QUESTION 1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most?
 
Figure 1. Distribution of food complexity by food type
Based on figure 1, the complexity ratings reveal clear distinctions between food types. Pizza shows the lowest complexity (mean 2.88), while Sushi rates highest (mean 3.42). Shawarma falls between these extremes (mean 3.23). These differences suggest preparation difficulty does varies across food categories.

 QUESTION 2. How many ingredients would you expect this food item to contain?
 
Figure 2. Distribution of ingredient counts by food type
The ingredient count distributions(figure 2) show distinct patterns across food types, approximating normal distributions with different parameters. Sushi requires the fewest ingredients (mean 4.96). Pizza shows an intermediate ingredient count (mean 6.30), while Shawarma demonstrates the highest ingredient complexity (mean 7.43). The difference in distributions makes this feature useful for discriminating between these food types in classification models.

QUESTION 3.  In what setting would you expect this food to be served? Please check all that apply.
 
Figure 3. Distribution of settings where food types are served
Based on figure 3, we can see significant difference in weekday_lunch, weekend_dinner, party and late_night. Shawarma is predominantly associated with weekday lunch (~90%). Pizza demonstrates exceptionally strong association with parties (~95%). Sushi exhibits strongest association with weekend dinner (~82%). The differences in when these foods are typically consumed provide useful signals for classification models to distinguish between these food categories.

QUESTION 4. How much would you expect to pay for one serving of this food item?

 
Figure 4: Expected price vs Food type
Based on figure 4, the expected price distributions also reveal significant differences between food types, making this a highly discriminative feature. Pizza exhibits the lowest price point (median $6.00). Shawarma shows a moderate price point (median $10.00) with the most consistent pricing. Sushi commands the highest prices (median $13.00). These distinct price distributions provide meaningful signals for classification, especially for distinguishing Sushi from Pizza.

QUESTION 5. What movie do you think of when thinking of this food item?
 
Figure 5. “Avengers” count by food type
The movie association data reveals an extraordinarily strong connection between Shawarma and the movie titles that contain the word “Avengers” (222 mentions) compared to minimal associations for Pizza (9) and Sushi (1), according to figure 5. The magnitude of this disparity suggests that text analysis of movie responses could reveal additional distinctive patterns for each food type. 

QUESTION 6. What drink would you pair with this food item?
 
Figure 6. Drink preferences by food type
The drink preferences show distinctive patterns that strongly differentiate food types(figure 6). Pizza demonstrates a very high relationship between cola (~45%) and soda (~24%). Similarly, sushi exhibits dramatically different preferences, with strong relationship with water (~34%) and the "other" category (~36%), while nearly avoiding carbonated drinks. Shawarma shows a more balanced distribution across beverage types. These pairing preferences provide excellent discriminative features for classification, particularly for distinguishing Pizza from Sushi, where preferences are almost complementary. 

QUESTION 7. When you think about this food item, who does it remind you of?
 
Figure 7: who each food type reminds people of
Based on figure 7, all three foods strongly associate with friends, but Pizza shows the highest friendship connection (~88%). Sushi demonstrates the strongest parental association (~50%). Shawarma uniquely shows a substantial association with strangers (~40%). Pizza maintains consistent associations across family members (parents, siblings), while Shawarma shows weaker family connections overall. In addition, pizza is associated to teachers much higher then sushi and shawarma. These patterns can provide meaningful classification signals by capturing the different social contexts in which these foods are experienced.

QUESTION 8.  How much hot sauce would you add to this food item?
 
Figure 8. Hot sauce preferences by food type
Figure 8 shows that hot sauce preferences reveal significant differences across food types. Sushi shows the strongest aversion to hot sauce, with approximately 70% of respondents preferring none. Shawarma demonstrates the opposite trend, with only about 15% choosing none and the majority preferring moderate to high amounts (approximately 60% combined). Pizza occupies a middle ground, with roughly half the respondents preferring none and the other half distributed across milder options. These distinctive spice preference patterns provide another valuable signal for classification models, particularly useful for distinguishing between Sushi and Shawarma.
