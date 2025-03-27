
help me modify clean_survey_data.py, 
such that, after finish processing the data as mentioned in the instructions, 

for question 2 and 4, 
1. calculate the median value amone the three groups, i. e sushi, shawarma, pizza.
Result: 
Q2: pizza 6.00, shawarma: 7.00, sushi: 5.00
Q4: 6.00, 10.00, 13.00
2. for all NULL values, assign the median value of that particular group to them

for quesion 5, 
for all NULL values, assign them a keyword "other"




QUESTION 1. From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most
 complex)
 1
 2
 3
 4
 5

 Cleaning logic: leave it as it is

 QUESTION 2. How many ingredients would you expect this food item to contain?

bad example: 632156,"I would expect this food item to contain at least five ingredients, taking a nigiri sushi as an example: rice, vinegar, sugar, salt, and fish.",Sushi

626771,"I would expect this food to contain multiple ingredients, but only a minimum of three elements. That means that it would take multiple ingredients to make a single element, i.e flour, salt, olive oil, yeast and other ingredients to make dough, where the dough is one element that is comprised of several ingredients. In terms of numbers, I would expect about 7-15 ingredients to make this food item.",Pizza

Cleaning logic: 
First, for every response, extract the integers from this response. 
case 1: this response contains only one integer. Then keep this integer
case 2: this response contains two integers, like 626771 above. Take the average between these two numbers. (to solve case like something - something)
case 3: This response contains more then two integers. This is too inconsistent, ignore this response and assign it an NULL value. 
case 4: this response don't even contain an integer, like 632156 above. Ignore this response and assign it an NULL value


QUESTION 3.  In what setting would you expect this food to be served? Please check all that apply
 Week day lunch
 Week day dinner
 Weekend lunch
 Weekend dinner
 At a party
 Late night snack

cleaning logic: processed into vector of 6*1

 QUESTION 4. How much would you expect to pay for one serving of this food item?
(Inconsistence answers, such as "A slice would probably be like $5, a whole pizza maybe $15?")

Cleaning logic: 
First, for every response, extract the integers from this response. 
case 1: this response contains only one integer. Then keep this integer
case 2: this response contains two integers. Take the average between these two numbers. (to solve case like 522890,8$-12$,Shawarma)
case 3: This response contains more then two integers. This is too inconsistent, ignore this response and assign it an NULL value. 
case 4: this response don't even contain an integer. Ignore this response and assign it an NULL value


QUESTION 5. What movie do you think of when thinking of this food item?
Bad example: 715814,"Not exactly movies, but I would prefer to watch some TV series when I am having pizzas. If movies specifically, I may think of cartoon movies or more ""hero""/marvel related ones, like Teenage Mutant Ninja Turtles, movies in marvel series, etc..",Pizza

526131,"I've never had them before attending U of T, so I don't associate them with anything.",Shawarma

Cleaning logic: Just keep all the response with length no more then 8 words. For longer ones, assign NULL to it. 
(note: i don't think this is a good predictor variable. super inconsistent. Could think of a better way to deal with this variable.)

QUESTION 6. What drink would you pair with this food item?

Cleaning logic: 
First, combine similar category into single term.
Combination logic: 
drink_groups = {
    'coca cola': ['coke', 'coca cola', 'diet coke', 'cola', 'coke zero', 'pepsi', 'diet pepsi'],
    'tea': ['tea', 'green tea', 'iced tea', 'ice tea', 'bubble tea', 'milk tea'],
    'water': ['water', 'sparkling water', 'mineral water', 'soda water'],
    'beer': ['beer', 'craft beer', 'cold beer'],
    'juice': ['juice', 'orange juice', 'apple juice', 'fruit juice', 'lemon juice'],
    'soda': ['soda', 'pop', 'soft drink', 'carbonated drink', 'sprite', 'fanta', '7up', 'mountain dew'],
    'wine': ['wine', 'red wine', 'white wine', 'rose wine'],
    'coffee': ['coffee', 'iced coffee', 'espresso', 'latte'],
    'milk': ['milk', 'chocolate milk', 'dairy']
}

Secondly, after combination, we will count the occurence of each category. Result shown in data/freeform_analysis/combined_top_drinks.csv

Then, take the top 4 categories, all the other drink become "other" category. For example coca cola is the top mentioned drink, and the response that contain coca cola will become a vector [1 0 0 0 0]. A response of soup will be [0 0 0 0 1] since soup is not contained within the top 4 drinks. 

QUESTION 7. When you think about this food item, who does it remind you of?
 Parents
 Siblings
 Friends
 Teachers
 Strangers

 Cleaning logic: Processed into vectors of 5*1

 QUESTION 8.  How much hot sauce would you add to this food item?
 None
 A little (mild)
 A moderate amount (medium)
 A lot (hot)
 I will have some of this food item with my hot sauce


Cleaning logic: 1 for None, 5 for I will have some of this food item with my hot sauce